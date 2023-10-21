import os
import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from g2p import pyopenjtalk_g2p_prosody
import commons
import soundfile as sf

def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference(config_path, G_model_path, text):
    device = "cpu"

    # load config.json
    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)  
    _ = net_g.eval()
    _ = utils.load_checkpoint(G_model_path, net_g, None)

    # parameter settings
    noise_scale     = torch.tensor(0.66)    # adjust z_p noise
    noise_scale_w   = torch.tensor(0.8)    # adjust SDP noise
    length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)

    # required_grad is False
    with torch.inference_mode():
        stn_phn = pyopenjtalk_g2p_prosody(text)
        stn_tst = get_text(stn_phn, hps)

        # generate audio
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst,
                            x_tst_lengths,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

    # save audio
    save_dir = os.path.join(os.getcwd(), "/infer_logs/")
    os.makedirs(save_dir, exist_ok=True)

    # Fixed filename
    save_path = os.path.join(save_dir, "output_audio.wav")
    try:
        sf.write(
             file=save_path,
             data=audio,
             samplerate=hps.data.sampling_rate,
             format="WAV")
    except Exception as e:
        # Handling file writing issues
        print(f"An error occurred while saving the audio file: {str(e)}")

    return 0

if __name__ == "__main__":
    config_path = "/Users/hk/Downloads/audio/config.json"
    G_model_path = "/Users/hk/Downloads/audio/G_simesaba.pth"
    text = "こんにちは"
    inference(config_path, G_model_path, text)