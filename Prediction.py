import torch
import json
from utils import beam_search, plot_attention_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(image_path, masked_image_path, saved_model_path, beam_size, device, saved_word_map_foler, att_map=False):
    # combine two img paths
    paths = [image_path, masked_image_path]

    # load id-words and words-ids pairs
    with open(saved_word_map_foler + 'id2word.json', 'r', encoding='utf-8') as f:
        id2word = json.load(f)
    with open(saved_word_map_foler + 'word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    id2word = {int(key): value for key, value in id2word.items()}

    # load trained model
    trained_model = torch.load(saved_model_path, map_location=device)

    # generate predicted report
    pred_report, probs = beam_search(trained_model, paths, word2id, id2word, beam_size)
    print('Predicted diagnosis report is: ', " ".join(pred_report))

    # plot attention map
    if att_map:
        plot_attention_map(pred_report, probs, masked_image_path)


def main():
    # parameters needed to generate report for a single image
    image_path = '../../autodl-tmp/medical_projects/archive/images/images_normalized/89_IM-2402-1001.dcm.png'
    masked_image_path = '../../autodl-tmp/medical_projects/Masked_Synthesized_Indiana_University_Dataset/Masked_89_IM-2402-1001.dcm.png'
    saved_word_map_folder = './'
    saved_model_path = './text_generation_best.pth'
    beam_size = 3
    plot_att_map = True  # whether plot the attention map or not

    predict(image_path, masked_image_path, saved_model_path, beam_size, device, saved_word_map_folder,
            att_map=plot_att_map)


if __name__ == "__main__":
    main()
