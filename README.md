<h2 align="center">
<p> :yum: Yui-mhcp :yum: </p>
</h2>

<h2 align="center">
<p> A centralization of Deep Learning projects </p>
</h2>


Welcome to my GitHub profile! The objective of my projects is to make Deep Learning more accessible, and provide real-world examples to enhance education and research in the field! :yum:

The goal is to aggregate a wide range of Deep Learning topics in one place with a common abstraction, making it easier for you to dive into this fascinating field. Each repository not only focuses on a specific area of Deep Learning, but also includes links to tutorials and reference papers. These resources are carefully selected to help you grasp both the practical and theoretical aspects of Deep Learning, which can sometimes be challenging to find.

## Latest release : `v1.1.0` (01/11/2026) :yum:

- The TensorRT-LLM `LLM API` is supported by the [LLM](https://github.com/yui-mhcp/language_models) and [STT](https://github.com/yui-mhcp/speech_to_text) projects : `huggingface` checkpoints are loaded directly, without any engine build step !
- New `Qwen 3.5` / `3.8` support, conversations with sessions, native + python tool calling and an experimental memory layer in the [LLM](https://github.com/yui-mhcp/language_models) project
- Data augmentation and an end-to-end `YOLO` training example in the [detection](https://github.com/yui-mhcp/detection) project
- `transformers 5` support of the `Tokenizer`, and many new tests

Check the [CHANGELOG](CHANGELOG.md) for the full list of changes, and the [installation guide](INSTALLATION.md) to set up the shared GPU environment !

## Covered topics

Currently developed topics: 
- [x] [Object/text detection](https://github.com/yui-mhcp/detection)
- [x] [Encoder networks](https://github.com/yui-mhcp/encoders)
- [x] [Large Language Models (LLM) / Natural Language Processing (NLP)](https://github.com/yui-mhcp/language_models)
- [x] [Optical Character Recognition (OCR)](https://github.com/yui-mhcp/ocr)
- [x] [Speech-To-Text (STT)](https://github.com/yui-mhcp/speech_to_text)
- [x] [Text-To-Speech (TTS)](https://github.com/yui-mhcp/text_to_speech)

Additional utilities: 
- [x] [Custom losses / metrics / callbacks](https://github.com/yui-mhcp/base_dl_project)
- [x] [Dataset processing / analyzing](https://github.com/yui-mhcp/data_processing)
- [x] [Plot / visualization utilities](https://github.com/yui-mhcp/data_processing)
- [x] [Audio processing](https://github.com/yui-mhcp/data_processing)
- [x] [Image processing](https://github.com/yui-mhcp/data_processing)
- [x] [Text processing](https://github.com/yui-mhcp/data_processing)
- [x] [Multi-threading framework](https://github.com/yui-mhcp/data_processing)

Practical use cases:
- [x] [Speaker Verification (SV)](https://github.com/yui-mhcp/encoders) (WIP : still being converted to keras 3)
- [x] [Information Retrieval (IR)](https://github.com/yui-mhcp/encoders)
- [x] [Keyword search in audios / videos](https://github.com/yui-mhcp/speech_to_text): tolerant search through the partial alignment of the transcription
- [x] [Retrieval-Augmented Generation (RAG) and tool calling](https://github.com/yui-mhcp/language_models): LLM answers grounded on your documents, with native and python tools
- [x] [Object detection training](https://github.com/yui-mhcp/detection): end-to-end `YOLOv2` training on a custom dataset
- [x] [Optical Character Recognition (OCR)](https://github.com/yui-mhcp/ocr): this project allows to detect text in an image, and performs OCR on the detected text

All topics are released in separate repositories to make it easier to learn / experiment with dedicated code and resources.


## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```