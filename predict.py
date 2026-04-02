from concurrent.futures import ThreadPoolExecutor
import torch
import esm
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import os
import torch.nn as nn
from pathlib import Path
from typing import List

prot_t5_model_path = Path("/data/zqwangyansu/hl/model/prot_t5_xl_uniref50")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import numpy


class FineTuneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        self.esm_converter = self.esm_alphabet.get_batch_converter()
        self.t5_tokenizer = T5Tokenizer.from_pretrained(prot_t5_model_path, do_lower_case=False)
        self.t5_model = T5EncoderModel.from_pretrained(prot_t5_model_path)
        

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280 + 1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1))
        self._unfreeze_layers()

    def _unfreeze_layers(self):
        for layer in self.esm_model.layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.t5_model.encoder.block[-2:]:
            for param in layer.parameters():
                param.requires_grad = False

    def _parallel_preprocess(self, batch):

        def process_esm(batch):
            esm_data = [(str(i), seq[:1024]) for i, seq in enumerate(batch)]
            _, _, tokens = self.esm_converter(esm_data)
            return tokens.to(device, non_blocking=True)

        def process_t5(batch):
            inputs = self.t5_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                return_attention_mask=True
            ).to(device)
            return inputs

        with ThreadPoolExecutor(max_workers=2) as executor:
            esm_future = executor.submit(process_esm, batch)
            t5_future = executor.submit(process_t5, batch)
            return esm_future.result(), t5_future.result()

    def forward(self, raw_sequences):
        esm_tokens, t5_inputs = self._parallel_preprocess(raw_sequences)

        with torch.cuda.amp.autocast():
            esm_output = self.esm_model(esm_tokens, repr_layers=[33])["representations"][33].mean(1)
            t5_output = self.t5_model(**t5_inputs).last_hidden_state.mean(1)
            fused = torch.cat([esm_output, t5_output], dim=1)
            return torch.sigmoid(self.classifier(fused))


def load_model(model_path: str) -> FineTuneModel:
    model = FineTuneModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    if device.type == 'cuda':
        model.half()
    return model

def predict(model,input,output_dir,sample_name):

    model = load_model(str(model))
    seqs = [str(rec.seq).replace("*", "") for rec in SeqIO.parse(input, "fasta")]
    headers = [rec.description for rec in SeqIO.parse(input, "fasta")]

    predictions = []
    batch_size = min(32,len(seqs))

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(batch)
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                predictions.extend(output.float().cpu().numpy())


    output_file = os.path.join(output_dir, f"{sample_name}.faa")
    saved_count = 0

    with open(output_file,'w') as f:
        for header, seq,pred in zip(headers,seqs,predictions):
            if pred > 0.90 :
                f.write(f">{header};prediction_score={pred}\n{seq}\n")
                saved_count += 1

'''
python predict.py \
    --model_path /data/zqwangyansu/hl/esm/plk/ARG-DPLM-MODEL.pth \
    --prot_t5_model_path /data/zqwangyansu/hl/model/prot_t5_xl_uniref50 \
    --input_dir /data/zqwangyansu/hl/human_gut_hdi/input_dir \
    --output_dir /data/zqwangyansu/hl/human_gut_hdi/output_dir \
    --batch_size 32 \
    --threshold 0.9
'''
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESM + ProtT5 High-confidence sequence screening")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the trained FineTune model")
    parser.add_argument("--prot_t5_model_path", type=str, required=True, help="ProtT5 Model Directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Input FASTA directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Predict batch size")
    parser.add_argument("--threshold", type=float, default=0.9, help="Filtering threshold")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    for file in input_dir.glob("*"):
        sample_name = file.stem
        if os.path.getsize(file) == 0:
            print(f"Skip empty file: {file}")
            continue
        predict_result = output_dir / f"{sample_name}.faa"
        if predict_result.exists():
            print(f"跳过已存在结果: {predict_result}")
            continue
        print(f"处理样本: {sample_name}")
        predict(args.model_path, file, output_dir, sample_name)


if __name__ == "__main__":
    main()



