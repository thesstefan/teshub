# Teshub

The goal of this project is to explore the task of manipulating weather in images using paired image-to-image translation. This problem has been specifically approached before by [WeatherGAN](https://arxiv.org/abs/2103.05422) through
unsupervised translation. The main hypothesis of the project is that images from fixed webcams with different weather conditions is a very good source of data for the paired translation and could yield better results than
unsupervised translation.

The name **Teshub** comes from the [Hurrian weather god](https://en.wikipedia.org/wiki/Teshub).

> This project is very experimental and is done as a learning exercise. Since generative models require a lot of resources (data, models and time),
> this is just a proof of concept, so the results are far from impressive. All models are trained on private data.

# What Is Weather & WeatherInFormer?

The WeatherInFormer multi-task model is based on [SegFormer](https://arxiv.org/abs/2105.15203) and its goal to "understand" weather in images.
The model tries to rate images based on 4 possible conditions: **snowy**, **cloudy**, **rainy** and **foggy**. To do this,
the model first segments weather-cues in the image and then uses them to take conclusions about the ratings.

## The 11 Weather Cues + Background
![image](https://github.com/thesstefan/teshub/assets/16565342/440c4549-0be3-40ce-9f5c-f70ee1fa1388)

## The WeatherInFormer Architecture
![image](https://github.com/thesstefan/teshub/assets/16565342/267ab6fa-dea2-4188-83ac-a78163454e42)

## WeatherInFormer Results
![image](https://github.com/thesstefan/teshub/assets/16565342/87cd2565-ebbf-423b-a5f2-961b7d9c86cc)

# How Does Weather Change & WeatherMorph?

The WeatherMorph is based on [Pix2Pix](https://arxiv.org/abs/1611.07004) and makes use of the information
obtained from *WeatherInFormer*. For each weather translation category, a specific configuration is chosen
that specifies the importance of each weather-cue. 

For training, the model is fed combinations of images from the same webcam/location, but with the
attributes specified by the translation configuration. For example, if we want to develop a model
that adds clouds, source images should have `cloudy < 0.2` and target images should have `cloudy > 0.8`.

## WeatherMoprh Architecture
![image](https://github.com/thesstefan/teshub/assets/16565342/ba3454ab-bbb0-4e27-ada9-6a7c29148ace)
![image](https://github.com/thesstefan/teshub/assets/16565342/cc007e17-1cb8-4a60-bf11-ae17139f7cb2)

## Weather Morph Results
![image](https://github.com/thesstefan/teshub/assets/16565342/188ae133-8284-47eb-a57f-ba1bf5ca8145)

# Teshub Website

A very simple website is developed to showcase the models (Flask backend + React frontend):
![image](https://github.com/thesstefan/teshub/assets/16565342/9d0dcb51-5bff-4528-aae0-567d80a4930a)

The website can be started locally by running `cd deploy && docker compose run` and then connecting in a 
browser to `localhost:5000`.
