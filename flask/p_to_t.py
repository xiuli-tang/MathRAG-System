
from pix2text import Pix2Text, merge_line_texts

p2t = Pix2Text.from_config()
def p_to_t(img_fp):
    outs = p2t.recognize_text_formula(img_fp, resized_shape=768, return_text=True)
    return outs

print(p_to_t("./img/img_22.png"))
