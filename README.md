<h2 aligne = "center">
<p> :yum: yui-mhcp :yum: </p>
</h2>

<h2 aligne = "center">
<p> A centralization of Deep Learning projects </p>
</h2>


- 👋 Hi, I’m @yui-mhcp ! :yum:
- 👀 I’m interested in practical usage of Artificial Intelligence (and more specifically Deep Learning)
- 🌱 I’m currently learning Computer Sciences
- 💞️ I’m looking to collaborate on Deep Learning projects on TTS, STT, NLP or any other funny topic ! :smile:
- 📫 You can ping me if you have any question, or send me an email / Discord message if you want to collaborate on a new project :smile:


The objective of this github is to propose many utilities to create, train, monitor and experiment models / architectures by tracking all configuration for reproducibility / comparison. 

Note that many projects / models already exist on GitHub (and some of my projects are highly inspired of some of them). The idea is to centralize all different topics, and provide abstract class\* / utilities to facilitate development, training, testing and deployment of (new) models. The major objective and benefit of such centralization and standardization is to allow an easy combination of multiple topics to create fancy applications !

\* The core of all projects is the `BaseModel` class described in [this project](https://github.com/yui-mhcp/base_dl_project) which is the base of all other projects.

## Type of projects

Currently developped topics : 
- [x] [Image classification](https://github.com/yui-mhcp/base_dl_project) \*
- [x] [Object detection](https://github.com/yui-mhcp/detection)
- [x] [Siamese networks](https://github.com/yui-mhcp/siamese_networks)
- [x] [Speech-To-Text (STT)](https://github.com/yui-mhcp/speech_to_text)
- [x] [Text-To-Speech (TTS)](https://github.com/yui-mhcp/text_to_speech)
- [x] [Generative Adversarial Networks (GAN)](https://github.com/yui-mhcp/generation)
- [x] [Object segmentation](https://github.com/yui-mhcp/detection)
- [ ] [Natural Language Processing (NLP) : Masked Language Modeling (MLM)](https://github.com/yui-mhcp/nlp)
- [x] [Natural Language Processing (NLP) : Question-Answering (Q&A)](https://github.com/yui-mhcp/nlp)
- [ ] [Natural Language Processing (NLP) : Text Generation](https://github.com/yui-mhcp/nlp)
- [ ] [Natural Language Processing (NLP) : Translation](https://github.com/yui-mhcp/nlp)
- [x] [Optical Character Recognition (OCR)](https://github.com/yui-mhcp/ocr)
- [ ] Reinforcment Learning (RL) for single-player games
- [ ] Reinforcment Learning (RL) for adversarial games

Additional utilities : 
- [x] [Custom losses / metrics / callbacks](https://github.com/yui-mhcp/base_dl_project)
- [x] [Dataset processing / analyzing](https://github.com/yui-mhcp/base_dl_project)
- [x] [Plot / visualization utils](https://github.com/yui-mhcp/data_processing)
- [x] [Audio processing](https://github.com/yui-mhcp/data_processing)
- [x] [Image processing](https://github.com/yui-mhcp/data_processing)
- [x] [Text processing](https://github.com/yui-mhcp/data_processing)
- [x] [Multi-threading framework](https://github.com/yui-mhcp/data_processing)

All topics are in separate repositories so that it is easier to contribute to a particular topic without carrying about other projects

\* It is a demonstration code to show how to subclass `BaseModel`. I will add a dedicated repository later for general classification (text / image / ...). 

## Objectives and applications

The main objective of this github is to democratize Deep Learning to facilitate its usage in **real-world projects**

The main idea is to **centralize many Deep Learning topics** to facilitate their learning. 
Furthermore, for each repository, I will try to put some **tutorial / references links** in order to give you some good tools to start this topic and learn theorical aspects of them (which can be quite hard to find for some of them :smile: ).  

I will also develop some applications / features with these repositories and will try to make them open-source as well (and always with free version) to give example of powerful, funny and helpful usage of Deep Learning even without big computational power !

If you create useful / funny concrete application with one of these project (or without), please contact me so I can reference them here ! :smile: It can give it more visibility and show example of real-world usage of Deep Learning methods 

If you have ideas of usage / interesting application but do not have time / experience to develop it, you can also contact me to add them and see if someone can help you to develop it

### Available features

- [x] [Speaker Verification (SV)](https://github.com/yui-mhcp/siamese_networks)
- [ ] [Face Recognition](https://github.com/yui-mhcp/siamese_networks) : the code is available but the model is not performant enough yet
- [x] [Search text in audios / videos](https://github.com/yui-mhcp/speech_to_text)
- [x] [Live transcription / subtitle generation](https://github.com/yui-mhcp/speech_to_text) : some models tend to be accurate enough for transcription but they will be further improved
- [x] [Text-To-Speech logger](https://github.com/yui-mhcp/text_to_speech) : `logging`-based logger that reads your logs with `TTS` models
- [x] [Optical Character Recognition (OCR)](https://github.com/yui-mhcp/ocr) : this projects allows to detect text in an image, and performs OCR on the detected text

### Available applications

### Application / features ideas

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Acknowledgments

Thanks to @Ananas120 for his contribution and sharing his implementation of `Transformers` architectures + his master thesis' code about Q&A !
