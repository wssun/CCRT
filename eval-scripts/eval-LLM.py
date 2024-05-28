# Import the os package
import os
import pandas as pd
import argparse
import torch
import pathlib
import requests
import base64

api_key = 'xxxxxx'
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

artist_styles = {
    "Andy Warhol": "Pop Art, Popular Culture Imagery, Silkscreen Prints",
    "Caravaggio": "Baroque Style, Strong Contrast of Light and Shadow, Religious and Mythological Subjects",
    "Cezanne": "Post-Impressionism, Pioneer of Cubism, Analysis of Color and Form",
    "Salvador Dali": "Surrealism, Dreamlike Imagery, Melting Objects",
    "Diego Rivera": "Mexican Muralism, Social Realism, Large-Scale Murals",
    "Edvard Munch": "Expressionism, Emotional Outpouring, Combination of Abstract and Figurative",
    "Frédéric Auguste Bartholdi": "Full of strong symbolic meaning, neoclassical style, magnificent",
    "Frida Kahlo": "Surrealism, Autobiographical Elements, Mexican Culture and Feminism",
    "Georges Seurat": "Pointillism, Optical Mixing, Detailed Urban Scenes",
    "Jackson Pollock": "Abstract Expressionism, Drip Technique, Freedom and Disorder",
    "Jan van Huysum": "Dutch Flower Paintings, Delicate and Refined Still Lifes",
    "Keith Haring": "Street Art, Graphic Symbols, Simple and Bold Lines",
    "Margaret Keane": "Big-Eyed Children Paintings, Pop Culture, Portraiture",
    "Max Ernst": "Dadaism, Surrealism, Collage and Dream Imagery",
    "Monet": "Impressionism, Light and Shadow Changes, Outdoor Painting",
    "Picasso": "has cubist style, full of freedom, pay attention to the emotional expression of color",
    "René Magritte": "Surrealism, Unusual Combination of Objects, Mysterious Imagery",
    "Richard Serra": "Minimalism, Sculpture, Large-Scale Metal Works",
    "Van Gogh": "has emotional passion colors, decorated with large color blocks of contrast"
}

base_path = "true_artwork"

def get_gpt4_answer(image0,image1,image2,eval_img, question_text):
    # Getting the base64 string
    base64_image0 = encode_image(image0)
    base64_image1 = encode_image(image1)
    base64_image2 = encode_image(image2)
    base64_eval_image = encode_image(eval_img)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{question_text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image0}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_eval_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        evaluation = result['choices'][0]['message']['content']
        return evaluation
    else:
        print(f'Error: {response.status_code}')

def evaluate_image_text(img0,img1,img2,image_path, question_text, epoch_id):
    epoch_id = str(epoch_id)
    file_names = sorted([file for file in image_path.glob(r'{}_*.png'.format(epoch_id))])
    answer = []
    file_names = file_names[:5]
    for image in file_names:
        res = str(get_gpt4_answer(img0,img1,img2,str(image),question_text))
        answer.append(res)
        print(f"image:{image}  gpt4：{res}")
    return answer


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='eval', description='Evaluate GPT4-Style-Score')
    parser.add_argument('--image_dir', help='dir of images', type=str, required=False,
                        default="compvis-word_Cezannestyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05_v_p_m_c_gpt4_entityV1_w001-em_0-et_1")
    parser.add_argument('--prompt_dir', help='dir of prompts', type=str, required=False,
                        default="data/art_prompts_gpt.csv")
    parser.add_argument('--save_dir', help='path to save results', type=str, required=False, default="RSR_multi")

    args = parser.parse_args()


    artist_list = ["Monet", "Cezanne", "Picasso", "Van Gogh", "Salvador Dali", "Andy Warhol",
                   "Edvard Munch", "Frédéric Auguste Bartholdi", "Diego Rivera", "Georges Seurat",
                   "Margaret Keane", "Jan van Huysum", "Frida Kahlo", "René Magritte", "Jackson Pollock",
                   "Richard Serra", "Max Ernst", "Keith Haring", "Caravaggio"]
    # artist_list = ["Van Gogh","Picasso","Monet"]
    image_dir = args.image_dir
    prompt_dir = args.prompt_dir

    path_image = pathlib.Path(image_dir)
    df_prompt = pd.read_csv(prompt_dir)

    image_names = os.listdir(image_dir)

    epoch_ids = sorted(set(int(name.split("_")[0]) for name in image_names if '.png' in name or '.jpg' in name))
    df = pd.DataFrame()
    epoch_ids.sort()


    eval_artist_list =  ["Van Gogh"]
    for eval_artist in eval_artist_list:
        eval_style = artist_styles[eval_artist]
        question = "The style of <artist> work is : <style>. " \
                   "As shown in the first three images. " \
                   "Please determine if the last picture is in one of the <artist_list> art styles. " \
                   "Just answer yes or no. If the painting has no artistic style, answer null."\
                    "The quality of some images may be poor, please do not misjudge"
        question = question.replace('<artist_list>', str(artist_list))
        question = question.replace('<artist>', str(eval_artist))
        question = question.replace('<style>', eval_style)
        img0 = f"{base_path}\\{str(eval_artist)}\\0.jpg"
        img1 = f"{base_path}\\{str(eval_artist)}\\1.jpg"
        img2 = f"{base_path}\\{str(eval_artist)}\\2.jpg"

        for epoch_id in epoch_ids:
            artist = df_prompt.iloc[int(epoch_id)]['artist']

            print(f"now eval: {eval_artist}")
            if artist == eval_artist:
                result_list = evaluate_image_text(img0, img1, img2, path_image, question, epoch_id)
                result_df = pd.DataFrame([result_list], columns=[f"{i}" for i in range(len(result_list))])
                df = pd.concat([df, result_df], ignore_index=True)

    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f"{args.save_dir}/{str(eval_artist_list)}.csv")
