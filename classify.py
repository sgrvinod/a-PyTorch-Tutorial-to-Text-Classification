import torch
from torch import nn
from utils import preprocess, rev_label_map
import json
import os
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = 'BEST_checkpoint_han.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Word map to encode with
data_folder = '/media/ssd/han data'
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def classify(document):
    """
    Classify a document with the Hierarchial Attention Network (HAN).

    :param document: a document in text form
    :return: pre-processed tokenized document, class scores, attention weights for words, attention weights for sentences, sentence lengths
    """
    # A list to store the document tokenized into words
    doc = list()

    # Tokenize document into sentences
    sentences = list()
    for paragraph in preprocess(document).splitlines():
        sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

    # Tokenize sentences into words
    for s in sentences[:sentence_limit]:
        w = word_tokenizer.tokenize(s)[:word_limit]
        if len(w) == 0:
            continue
        doc.append(w)

    # Number of sentences in the document
    sentences_in_doc = len(doc)
    sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)  # (1)

    # Number of words in each sentence
    words_in_each_sentence = list(map(lambda s: len(s), doc))
    words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                 words_in_each_sentence)  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim=0)  # (n_classes)
    word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    words_in_each_sentence = words_in_each_sentence.squeeze(0)  # (n_sentences)

    return doc, scores, word_alphas, sentence_alphas, words_in_each_sentence


def visualize_attention(doc, scores, word_alphas, sentence_alphas, words_in_each_sentence):
    """
    Visualize important sentences and words, as seen by the HAN model.

    :param doc: pre-processed tokenized document
    :param scores: class scores, a tensor of size (n_classes)
    :param word_alphas: attention weights of words, a tensor of size (n_sentences, max_sent_len_in_document)
    :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)
    """
    # Find best prediction
    score, prediction = scores.max(dim=0)
    prediction = '{category} ({score:.2f}%)'.format(category=rev_label_map[prediction.item()], score=score.item() * 100)

    # For each word, find it's effective importance (sentence alpha * word alpha)
    alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(
        1).float() / words_in_each_sentence.max().float())
    # alphas = word_alphas * words_in_each_sentence.unsqueeze(1).float() / words_in_each_sentence.max().float()
    alphas = alphas.to('cpu')

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word, because size is scaled by normalized word*sentence alphas
    max_font_size = 55  # maximum size possible for a word, because size is scaled by normalized word*sentence alphas
    space_size = ImageFont.truetype("./calibril.ttf", max_font_size).getsize(' ')  # use spaces of maximum font size
    line_spacing = 15  # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    max_rectangle_width = 0.8 * left_buffer  # maximum width of the rectangles that will represent sentence alphas, scaled by sentence alpha
    rectangle_loc = [0.9 * left_buffer,
                     image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = list()
    sentence_viz_properties = list()
    for s, sentence in enumerate(doc):
        # Find visualization properties for each sentence, represented by rectangles
        # Factor to scale by
        sentence_factor = sentence_alphas[s].item() / sentence_alphas.max().item()

        # Color of rectangle
        rectangle_saturation = str(int(sentence_factor * 100))
        rectangle_lightness = str(25 + 50 - int(sentence_factor * 50))
        rectangle_color = 'hsl(0,' + rectangle_saturation + '%,' + rectangle_lightness + '%)'

        # Bounds of rectangle
        rectangle_bounds = [rectangle_loc[0] - sentence_factor * max_rectangle_width,
                            rectangle_loc[1] - rectangle_height] + rectangle_loc

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})

        for w, word in enumerate(sentence):
            # Find visualization properties for each word
            # Factor to scale by
            word_factor = alphas[s, w].item() / alphas.max().item()

            # Color of word
            word_saturation = str(int(word_factor * 100))
            word_lightness = str(25 + 50 - int(word_factor * 50))
            word_color = 'hsl(0,' + word_saturation + '%,' + word_lightness + '%)'

            # Size of word
            word_font_size = int(min_font_size + word_factor * (max_font_size - min_font_size))
            word_font = ImageFont.truetype("./calibril.ttf", word_font_size)

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = word_font.getsize(word)
            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.text(xy=viz['loc'], text=viz['word'], fill=viz['color'], font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])
    # Detected category/topic
    category_font = ImageFont.truetype("./calibril.ttf", min_font_size)
    draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    draw.text(xy=[line_spacing, line_spacing + category_font.getsize('Detected Category:')[1] + line_spacing],
              text=prediction.upper(), fill='black',
              font=category_font)
    del draw

    # Display
    img.show()



if __name__ == '__main__':
    document = 'How do computers work? I have a CPU I want to use. But my keyboard and motherboard do not help.\n\n You can just google how computers work. Honestly, its easy.'
    document = 'But think about it! It\'s so cool. Physics is really all about math. what feynman said, hehe'
    document = "I think I'm falling sick. There was some indigestion at first. But now a fever is beginning to take hold."
    document = "I want to tell you something important. Get into the stock market and investment funds. Make some money so you can buy yourself some yogurt."
    document = "You know what's wrong with this country? republicans and democrats. always at each other's throats\n There's no respect, no bipartisanship."
    visualize_attention(*classify(document))
