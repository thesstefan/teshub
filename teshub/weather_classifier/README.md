# Weather Classification & Segmentation

This model is used to infer the segmentation mask and weather labels of images (in our case from streams). 
The code is an adaptation of https://github.com/wzgwzg/Multiask_Weather, tuned especially for
inference, in the case of clear/sunny classification and segmentation. 

A pre-trained model from
[Baidu](https://pan.baidu.com/s/1pMDE2uv),
provided by the authors of
[Weather Recognition via Classification Labels and Weather-Cue Maps](https://sci-hub.se/https://www.sciencedirect.com/science/article/pii/S0031320319302481)
is used It's also uploaded on 
[GDrive](https://drive.google.com/file/d/1q7OjUZgz2ZzPzfsHPzbNluJye450oid9/view?usp=share_link)
for easier access (downloading from Baidu is very hard for people that are not Chinese).

![image](https://user-images.githubusercontent.com/16565342/206379692-6b73b1d1-1e9c-4016-94f6-dbfb2cae26d1.png)

## Setup

The code uses TFv1 and Python2, so the user must set up a virtual environment with `python2.7` and install 
the dependencies provided in the `requirements.txt` file. For example, using `virtualenv` this can be done 
by running the following lines:
```bash
virtualenv -p /usr/bin/python2.7 env
source env/bin/activate
pip install -r requirements.txt
```

Now, provided that the user has input images in the `data` directory, one can run the inference using
```bash
python run_inference.py --input_dir data
```

Details about the predictions are then available in the `predictions.json` file. For more details, check
the [paper](https://github.com/thesstefan/vae_weather_translation/blob/build/vae_weather_translation.pdf).

Note that the program will attempt to download the model data from GDrive in the `model` directory if
it's not already present.
