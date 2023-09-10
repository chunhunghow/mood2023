import os
import glob
import nibabel as nib
import numpy as np
import time
from mood_docker import *
import torch

def predict_folder_pixel_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):

        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()
        nimg_array = nimg_array.transpose(2,1,0)


        #nimg_array[nimg_array < 0.01] = 0.5

        #abnomal_score_array = np.abs(nimg_array - 0.5)

        final_nimg = nib.Nifti1Image(abnomal_score_array, affine=nimg.affine)
        nib.save(final_nimg, target_file)


def predict_folder_sample_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):

        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()
        nimg = nimg.transpose(2,1,0)
        #abnomal_score = np.random.rand()
        score = model.predict(nimg_array)


        with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
            write_file.write(str(abnomal_score))


def predict_folder_abs(input_folder, target_folder, mode):
    for f in os.listdir(input_folder):

        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()
        nimg_array = nimg_array.transpose(2,1,0)
        #abnomal_score = np.random.rand()

        start = time.time()
        abnormal_score,mask = model.predict(nimg_array)
        print('Taking ',time.time() - start, 's')

        mask = mask.transpose(-1,1,0)

        if mode == 'sample':
            with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
                write_file.write(str(abnormal_score))

        else:
            final_nimg = nib.Nifti1Image(mask, affine=nimg.affine)
            nib.save(final_nimg, target_file)






if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--modality", type=str, help="can be either 'brain' or 'abdom'.", required=True)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    assert args.modality in ['brain','abdom'] , 'Must be either brain or abdom'
    with open('/workspace/stats.yaml','r') as stream:
    #with open('stats.yaml','r') as stream:
        crt_val = yaml.safe_load(stream)

    threshold = crt_val[args.modality]


    model = Model(model='vitvae', 
                  _3d=False,
                  max_batch_size=32,
                  threshold = threshold
                  )
    ckpt = glob.glob(f'/workspace/weights/{args.modality}/*')[0]
    #ckpt = glob.glob(f'weights/{args.modality}/*')[0]

    model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu'))['state_dict'])
    print('Loaded weights successfully.')

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


    #if mode == "pixel":
    #    predict_folder_pixel_abs(input_dir, output_dir)
    #elif mode == "sample":
    #    predict_folder_sample_abs(input_dir, output_dir)
    #else:
    #    print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

    predict_folder_abs(input_dir, output_dir, mode)

    # predict_folder_sample_abs("/home/david/data/datasets_slow/mood_brain/toy", "/home/david/data/datasets_slow/mood_brain/target_sample")
