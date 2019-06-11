# X-Punctuator
A PyTorch implementation of punctuation prediction system using LSTM/BLSTM [1][2][3], which automatically adds suitable punctation into text without punctuation.

## Install
- PyTorch 0.4+

## Usage
`egs/toy/run.sh` provides an example usage.
```bash
# Set PATH and PYTHONPATH
$ cd egs/toy/; . ./path.sh
# Train
$ train.py -h
# Add punctuation
$ add_punctuation.py -h
# Analyze metrics
$ analyer.py -h
```

#### How to visualize loss?
If you want to visualize your loss, you can use [visdom](https://github.com/facebookresearch/visdom) to do that:
1. Open a new terminal in your remote server (recommend tmux) and run `$ visdom`.
2. Open a new terminal and run `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`.
3. Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`.
4. In visdom website, chose `<any-string>` in `Environment` to see your loss.
#### How to resume training?
```bash
$ train.py --continue_from <model-path>
```
#### How to use multi-GPU?
Use comma separated gpu-id sequence, such as:
```bash
$ CUDA_VISIBLE_DEVICES="0,1" train.py
```

## Reference
- [1] Kaituo Xu, Lei Xie, and Kaisheng Yao. "Investigating LSTM for punctuation prediction" in ISCSLP 2016
- [2] Ottokar Tilk and Tanel Alumae. "Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration" in Interspeech 2016
- [3] Ottokar Tilk and Tanel Alumae. "LSTM for Punctuation Restoration in Speech Transcripts" in Interspeech 2015
