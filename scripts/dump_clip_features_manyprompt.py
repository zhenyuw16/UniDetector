# Copyright (c) Facebook, Inc. and its affiliates.
import pickle
import argparse
import json
import torch
import numpy as np
import itertools
import sys

if __name__ == '__main__':

    prompt_templates = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]


    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model', default="ViT-B/32")
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in sorted(data['categories'], key=lambda x: x['id'])]
    cat_names = [x.replace(',', '').replace('+', ' ') for x in cat_names]

    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_sentences = []
    for i in range(len(cat_names)):
        cat_name = cat_names[i]
        sentences = [prompt.format(cat_name) for prompt in prompt_templates]
        all_sentences.append(sentences)


    import clip
    print('Loading CLIP')
    model, preprocess = clip.load(args.clip_model, device=device)
    all_text_features = []
    for i in range(len(all_sentences)):
        text = clip.tokenize(all_sentences[i]).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = model.encode_text(text)
        all_text_features.append(torch.mean(text_features, 0, keepdims=True))
    all_text_features = torch.cat(all_text_features)
    print('text_features.shape', all_text_features.shape)
    all_text_features = all_text_features.cpu().numpy()
    
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), all_text_features)
