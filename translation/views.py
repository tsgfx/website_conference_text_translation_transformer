from django.shortcuts import render
from django.http import JsonResponse
import torch
import re
from pathlib import Path
from sacremoses import MosesTokenizer, MosesDetokenizer
from transformer_bpe_50000_large import TransformerModel as TransformerModelEn2Zh
from transformer_bpe_50000_large_zh2en import TransformerModel as TransformerModelZh2En
from transformer_bpe_50000_large import Tokenizer
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
from django.utils.html import escape

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径
EN2ZH_MODEL_PATH = "./checkpoints/model_bpe_50000_large_epoch40_not_share/best.ckpt"
ZH2EN_MODEL_PATH = "./checkpoints/model_bpe_1000000_large_epoch40_zh2en_not_share/best.ckpt"
EN2ZH_DATASET_PATH = "./train_data_size_1000000"
ZH2EN_DATASET_PATH = "./train_data_size_1000000_zh2en"

class Translator:
    def __init__(self, model_en2zh, model_zh2en, en2zh_en_tokenizer, en2zh_zh_tokenizer, zh2en_zh_tokenizer, zh2en_en_tokenizer):
        self.model_en2zh = model_en2zh
        self.model_zh2en = model_zh2en
        self.model_en2zh.eval()
        self.model_zh2en.eval()
        self.en2zh_en_tokenizer = en2zh_en_tokenizer
        self.en2zh_zh_tokenizer = en2zh_zh_tokenizer
        self.zh2en_zh_tokenizer = zh2en_zh_tokenizer
        self.zh2en_en_tokenizer = zh2en_en_tokenizer
        self.mose_tokenizer_en = MosesTokenizer(lang="en")
        self.mose_detokenizer_zh = MosesDetokenizer(lang="zh")
        self.mose_tokenizer_zh = MosesTokenizer(lang="zh")
        self.mose_detokenizer_en = MosesDetokenizer(lang="en")
        self.pattern = re.compile(r'(@@ )|(@@ ?$)')

    def translate(self, sentence, mode="en2zh"):
        try:
            if mode == "en2zh":
                # 英译中处理
                tokenized_sentence = " ".join(self.mose_tokenizer_en.tokenize(sentence.lower()))
                logger.info(f"[英译中] Moses分词结果: {tokenized_sentence}")
                
                encoder_input, attn_mask = self.en2zh_en_tokenizer.encode([tokenized_sentence.split()], add_bos=True, add_eos=True, return_mask=True)
                logger.info(f"[英译中] BPE编码结果: {encoder_input}")
                
                encoder_input = torch.Tensor(encoder_input).to(dtype=torch.int64)
                outputs = self.model_en2zh.infer(encoder_inputs=encoder_input, encoder_inputs_mask=attn_mask)
                preds = outputs.preds.numpy()
                logger.info(f"[英译中] 模型输出ID: {preds}")
                
                decoded = self.en2zh_zh_tokenizer.decode(preds)[0]
                logger.info(f"[英译中] BPE解码结果: {decoded}")
                
                final_result = self.mose_detokenizer_zh.tokenize(self.pattern.sub("", decoded).split())
                logger.info(f"[英译中] 最终结果: {final_result}")
                return final_result
                
            else:  # zh2en
                # 中译英处理
                # 首先检查输入的中文字符
                logger.info(f"[中译英] 原始输入: {sentence}")
                
                # 对中文进行字符级分词
                chars = list(sentence)
                tokenized_sentence = " ".join(chars)
                logger.info(f"[中译英] 字符分词结果: {tokenized_sentence}")
                
                # 使用 Moses 分词器
                moses_tokens = self.mose_tokenizer_zh.tokenize(tokenized_sentence)
                tokenized_sentence = " ".join(moses_tokens)
                logger.info(f"[中译英] Moses分词结果: {tokenized_sentence}")
                
                # BPE 编码前检查词表
                logger.info(f"[中译英] 词表大小: {len(self.zh2en_zh_tokenizer.word2idx)}")
                logger.info(f"[中译英] 分词后的tokens: {tokenized_sentence.split()}")
                
                # BPE 编码
                encoder_input, attn_mask = self.zh2en_zh_tokenizer.encode(
                    [tokenized_sentence.split()], 
                    add_bos=True, 
                    add_eos=True, 
                    return_mask=True
                )
                logger.info(f"[中译英] BPE编码结果: {encoder_input}")
                
                # 转换为 tensor
                encoder_input = torch.Tensor(encoder_input).to(dtype=torch.int64)
                
                # 模型推理
                outputs = self.model_zh2en.infer(encoder_inputs=encoder_input, encoder_inputs_mask=attn_mask)
                preds = outputs.preds.numpy()
                logger.info(f"[中译英] 模型输出ID: {preds}")
                
                # 解码
                decoded = self.zh2en_en_tokenizer.decode(preds)[0]
                logger.info(f"[中译英] BPE解码结果: {decoded}")
                
                # 去除 BPE 标记
                cleaned = self.pattern.sub("", decoded)
                logger.info(f"[中译英] 清理BPE标记后: {cleaned}")
                
                # Moses 去分词
                detokenized = self.mose_detokenizer_en.tokenize(cleaned.split())
                logger.info(f"[中译英] Moses去分词结果: {detokenized}")
                
                # 最终处理
                final_result = str(detokenized).capitalize()
                logger.info(f"[中译英] 最终结果: {final_result}")
                return final_result
                
        except Exception as e:
            logger.error(f"翻译错误: {str(e)}")
            logger.error(f"模式: {mode}")
            logger.error(f"输入: {sentence}")
            logger.error("错误追踪:", exc_info=True)
            raise e

def init_translator():
    base_config = {
        "bos_idx": 1,
        "eos_idx": 3,
        "pad_idx": 0,
        "max_length": 128,
        "d_model": 512,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "layer_norm_eps": 1e-6,
        "num_heads": 8,
        "num_decoder_layers": 6,
        "num_encoder_layers": 6,
        "label_smoothing": 0.1,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1e-9,
        "warmup_steps": 4000,
    }

    # 加载英译中词表
    en2zh_en_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    en2zh_zh_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    en2zh_en_idx2word = {v: k for k, v in en2zh_en_word2idx.items()}
    en2zh_zh_idx2word = {v: k for k, v in en2zh_zh_word2idx.items()}
    
    # 加载中译英词表
    zh2en_zh_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    zh2en_en_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    zh2en_zh_idx2word = {v: k for k, v in zh2en_zh_word2idx.items()}
    zh2en_en_idx2word = {v: k for k, v in zh2en_en_word2idx.items()}
    
    threshold = 1

    # 加载英译中词表
    with open(f"{EN2ZH_DATASET_PATH}/en.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                en2zh_en_word2idx[token] = len(en2zh_en_word2idx)
                en2zh_en_idx2word[len(en2zh_en_idx2word)] = token

    with open(f"{EN2ZH_DATASET_PATH}/zh.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                en2zh_zh_word2idx[token] = len(en2zh_zh_word2idx)
                en2zh_zh_idx2word[len(en2zh_zh_idx2word)] = token

    # 加载中译英词表
    with open(f"{ZH2EN_DATASET_PATH}/zh.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                zh2en_zh_word2idx[token] = len(zh2en_zh_word2idx)
                zh2en_zh_idx2word[len(zh2en_zh_idx2word)] = token

    with open(f"{ZH2EN_DATASET_PATH}/en.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                zh2en_en_word2idx[token] = len(zh2en_en_word2idx)
                zh2en_en_idx2word[len(zh2en_en_idx2word)] = token

    # 创建英译中配置
    en2zh_config = base_config.copy()
    en2zh_config["en_vocab_size"] = len(en2zh_en_word2idx)
    en2zh_config["zh_vocab_size"] = len(en2zh_zh_word2idx)

    # 创建中译英配置
    zh2en_config = base_config.copy()
    zh2en_config["en_vocab_size"] = len(zh2en_en_word2idx)
    zh2en_config["zh_vocab_size"] = len(zh2en_zh_word2idx)

    # 创建tokenizer
    en2zh_en_tokenizer = Tokenizer(en2zh_en_word2idx, en2zh_en_idx2word)
    en2zh_zh_tokenizer = Tokenizer(en2zh_zh_word2idx, en2zh_zh_idx2word)
    zh2en_en_tokenizer = Tokenizer(zh2en_en_word2idx, zh2en_en_idx2word)
    zh2en_zh_tokenizer = Tokenizer(zh2en_zh_word2idx, zh2en_zh_idx2word)

    # 加载英译中模型
    model_en2zh = TransformerModelEn2Zh(en2zh_config)
    model_en2zh.load_state_dict(torch.load(EN2ZH_MODEL_PATH, map_location="cpu", weights_only=True))
    model_en2zh.eval()

    # 加载中译英模型
    model_zh2en = TransformerModelZh2En(zh2en_config)
    model_zh2en.load_state_dict(torch.load(ZH2EN_MODEL_PATH, map_location="cpu", weights_only=True))
    model_zh2en.eval()

    return Translator(
        model_en2zh, 
        model_zh2en, 
        en2zh_en_tokenizer, 
        en2zh_zh_tokenizer,
        zh2en_zh_tokenizer,
        zh2en_en_tokenizer
    )

translator = init_translator()

def translate_text(request):
    if request.method == "POST":
        try:
            text = request.POST.get("text", "").strip()
            mode = request.POST.get("mode", "en2zh")
            
            logger.info(f"当前模式: {'英译中' if mode == 'en2zh' else '中译英'}")
            logger.info(f"输入文本: {text}")
            
            if not text:
                error_msg = "输入文本不能为空"
                # 如果是AJAX请求，返回JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({"error": error_msg, "mode": mode})
                
                return render(request, "index.html", {
                    "error": error_msg,
                    "mode": mode,
                    "text": text
                })

            if len(text) > 2000:
                error_msg = "输入文本过长，请限制在2000字符以内"
                # 如果是AJAX请求，返回JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({"error": error_msg, "mode": mode, "text": text})
                
                return render(request, "index.html", {
                    "error": error_msg,
                    "mode": mode,
                    "text": text
                })

            try:
                translated_text = translator.translate(text, mode)
                logger.info(f"翻译结果: {translated_text}")
                
                # 转义特殊字符
                safe_text = escape(text)
                safe_translated_text = escape(translated_text)
                
                # 如果是AJAX请求，返回JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        "translated_text": safe_translated_text,
                        "text": safe_text,
                        "mode": mode
                    })
                
                # 否则返回HTML
                return render(request, "index.html", {
                    "translated_text": safe_translated_text,
                    "text": safe_text,
                    "mode": mode
                })
                
            except Exception as e:
                logger.error(f"翻译过程出错: {str(e)}")
                error_msg = "翻译服务出现错误，请稍后重试"
                
                # 如果是AJAX请求，返回JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        "error": error_msg,
                        "text": text,
                        "mode": mode
                    }, status=500)
                    
                return render(request, "index.html", {
                    "error": error_msg,
                    "text": text,
                    "mode": mode
                })

        except Exception as e:
            logger.error(f"服务器错误: {str(e)}")
            error_msg = "服务器内部错误"
            
            # 如果是AJAX请求，返回JSON
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    "error": error_msg,
                    "mode": mode
                }, status=500)
                
            return render(request, "index.html", {
                "error": error_msg,
                "mode": mode
            })

    return render(request, "index.html")

def index(request):
    return render(request, "index.html")
