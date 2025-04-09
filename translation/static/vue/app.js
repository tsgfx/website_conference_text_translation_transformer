const { createApp, ref, computed, watch, onMounted } = Vue;

// 翻译头部组件
const AppHeader = {
    template: `
        <header class="header">
            <h1><i class="fas fa-language"></i> 智能翻译系统</h1>
            <p class="subtitle">基于深度学习的高精度翻译服务</p>
        </header>
    `
};

// 语言切换组件
const LanguageSwitch = {
    props: ['modelValue'],
    emits: ['update:modelValue'],
    template: `
        <div class="language-switch">
            <button type="button" :class="['switch-button', modelValue === 'en2zh' ? 'active' : '']" 
                   @click="$emit('update:modelValue', 'en2zh')">
                <i class="fas fa-arrow-right"></i> 英 → 中
            </button>
            <span class="direction-icon">|</span>
            <button type="button" :class="['switch-button', modelValue === 'zh2en' ? 'active' : '']" 
                   @click="$emit('update:modelValue', 'zh2en')">
                <i class="fas fa-arrow-left"></i> 中 → 英
            </button>
        </div>
    `
};

// 输入区域组件
const InputArea = {
    props: ['modelValue', 'mode'],
    emits: ['update:modelValue'],
    computed: {
        placeholder() {
            return this.mode === 'en2zh' ? '请输入要翻译的英文内容...' : '请输入要翻译的中文内容...';
        },
        inputLabel() {
            return this.mode === 'en2zh' ? '英文输入' : '中文输入';
        },
        textClass() {
            return this.mode === 'en2zh' ? 'en-text' : 'zh-text';
        },
        charCount() {
            return this.modelValue ? this.modelValue.length : 0;
        }
    },
    methods: {
        clearText() {
            this.$emit('update:modelValue', '');
        },
        copyText() {
            navigator.clipboard.writeText(this.modelValue).then(() => {
                this.$emit('show-toast', '复制成功！');
            }).catch(() => {
                this.$emit('show-toast', '复制失败，请手动复制');
            });
        }
    },
    template: `
        <div class="translation-box input-box">
            <div class="box-header">
                <h3><i class="fas fa-pen"></i> {{ inputLabel }}</h3>
                <div class="tools">
                    <button type="button" class="tool-button" @click="clearText">
                        <i class="fas fa-eraser"></i>
                    </button>
                    <button type="button" class="tool-button" @click="copyText">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </div>
            <textarea 
                :class="textClass"
                :placeholder="placeholder"
                :value="modelValue"
                @input="$emit('update:modelValue', $event.target.value)"
                rows="4"></textarea>
            <div class="char-count">字符数：<span>{{ charCount }}</span></div>
        </div>
    `
};

// 输出区域组件
const OutputArea = {
    props: ['translatedText', 'error', 'mode'],
    emits: ['show-toast'],
    computed: {
        outputLabel() {
            return this.mode === 'en2zh' ? '中文翻译' : '英文翻译';
        },
        textClass() {
            return this.mode === 'en2zh' ? 'output-content zh-text' : 'output-content en-text';
        }
    },
    methods: {
        copyText() {
            const text = this.translatedText || '';
            navigator.clipboard.writeText(text).then(() => {
                this.$emit('show-toast', '复制成功！');
            }).catch(() => {
                this.$emit('show-toast', '复制失败，请手动复制');
            });
        },
        // 新增：导出为TXT文件功能
        exportToTxt() {
            if (!this.translatedText) {
                this.$emit('show-toast', '没有可导出的翻译结果');
                return;
            }

            // 创建Blob对象
            const blob = new Blob([this.translatedText], { type: 'text/plain;charset=utf-8' });

            // 创建下载链接
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');

            // 设置文件名，使用当前日期时间作为文件名
            const now = new Date();
            const fileName = `翻译结果_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}.txt`;

            a.href = url;
            a.download = fileName;

            // 触发下载
            document.body.appendChild(a);
            a.click();

            // 清理
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                this.$emit('show-toast', '导出成功！');
            }, 100);
        }
    },
    template: `
        <div class="translation-box output-box">
            <div class="box-header">
                <h3><i class="fas fa-language"></i> {{ outputLabel }}</h3>
                <div class="tools">
                    <button type="button" class="tool-button" @click="copyText" title="复制文本">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button type="button" class="tool-button export-button" @click="exportToTxt" title="导出为TXT">
                        <i class="fas fa-file-download"></i>
                    </button>
                </div>
            </div>
            <div :class="textClass">
                <template v-if="translatedText">{{ translatedText }}</template>
                <span v-if="error" class="error"><i class="fas fa-exclamation-circle"></i> {{ error }}</span>
            </div>
        </div>
    `
};

// 翻译按钮组件
const TranslateButton = {
    template: `
        <div class="translate-button-container">
            <button type="submit" id="translateBtn">
                <i class="fas fa-arrow-right"></i>
            </button>
        </div>
    `
};

// 页脚组件
const AppFooter = {
    template: `
        <footer class="footer">
            <div class="features">
                <div class="feature">
                    <i class="fas fa-bolt"></i>
                    <span>快速翻译</span>
                </div>
                <div class="feature">
                    <i class="fas fa-check-circle"></i>
                    <span>准确度高</span>
                </div>
                <div class="feature">
                    <i class="fas fa-shield-alt"></i>
                    <span>安全可靠</span>
                </div>
            </div>
        </footer>
    `
};

// 提示框组件
const Toast = {
    props: ['message', 'isVisible', 'progress'],
    template: `
        <div class="toast" :class="{ 'show': isVisible }">
            {{ message }}
            <div v-if="progress > 0 && progress < 100" class="toast-progress">
                <div class="progress-bar" :style="{ width: progress + '%' }"></div>
            </div>
        </div>
    `
};

// 主应用组件
const App = {
    components: {
        AppHeader,
        LanguageSwitch,
        InputArea,
        OutputArea,
        TranslateButton,
        AppFooter,
        Toast
    },
    setup() {
        // 从HTML属性中获取初始数据
        const appElement = document.getElementById('app');

        const initialText = appElement ? appElement.getAttribute('data-text') || '' : '';
        const initialTranslatedText = appElement ? appElement.getAttribute('data-translated-text') || '' : '';
        const initialMode = appElement ? appElement.getAttribute('data-mode') || 'en2zh' : 'en2zh';
        const initialError = appElement ? appElement.getAttribute('data-error') || '' : '';

        const text = ref(initialText);
        const translatedText = ref(initialTranslatedText);
        const mode = ref(initialMode);
        const error = ref(initialError);
        const toastMessage = ref('');
        const toastVisible = ref(false);
        const progress = ref(0);

        // 更新分割文本为句子的函数
        const splitTextIntoSentences = (text) => {
            // 匹配中英文的句号、问号、感叹号、逗号、分号等标点
            const sentencePattern = /[.。,，;；！？!?]/g;

            // 分割文本
            const sentences = [];
            let lastIndex = 0;
            let match;

            while ((match = sentencePattern.exec(text)) !== null) {
                // 包含标点在内的句子
                const sentence = text.substring(lastIndex, match.index + 1).trim();
                if (sentence) sentences.push(sentence);
                lastIndex = match.index + 1;
            }

            // 添加最后一部分，如果没有以标点结尾
            const remaining = text.substring(lastIndex).trim();
            if (remaining) sentences.push(remaining);

            // 如果没有找到任何标点，则整个文本作为一个句子
            if (sentences.length === 0 && text.trim()) {
                sentences.push(text.trim());
            }

            // 过滤掉过短的句子，或者将它们与其他句子合并
            const filteredSentences = [];
            let currentSentence = '';

            for (const sentence of sentences) {
                // 如果句子过短（小于3个字符）或只包含标点符号，则与下一句合并
                if (sentence.length <= 3 || sentence.match(/^[.。,，;；！？!?\s]+$/)) {
                    currentSentence += sentence;
                } else {
                    // 如果有累积的短句，则添加到当前句子
                    if (currentSentence) {
                        currentSentence += sentence;
                        filteredSentences.push(currentSentence);
                        currentSentence = '';
                    } else {
                        filteredSentences.push(sentence);
                    }
                }
            }

            // 添加最后一个累积的句子（如果有）
            if (currentSentence) {
                filteredSentences.push(currentSentence);
            }

            return filteredSentences.length > 0 ? filteredSentences : sentences;
        };

        // 新增：翻译单个句子的函数
        const translateSentence = async (sentence, mode, csrftoken) => {
            // 创建表单数据
            const formData = new FormData();
            formData.append('text', sentence);
            formData.append('mode', mode);

            // 发送翻译请求
            const response = await fetch('/translate/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrftoken,
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (response.headers.get('content-type').includes('application/json')) {
                // 处理JSON响应
                const data = await response.json();
                if (data.translated_text) {
                    return data.translated_text;
                } else if (data.error) {
                    throw new Error(data.error);
                }
            } else {
                // 处理HTML响应
                const htmlText = await response.text();

                // 使用DOM操作提取翻译结果
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = htmlText;

                const extractedTranslation = tempDiv.querySelector('.output-content') ?
                    tempDiv.querySelector('.output-content').textContent.trim() : '';

                const extractedError = tempDiv.querySelector('.error') ?
                    tempDiv.querySelector('.error').textContent.trim() : '';

                if (extractedTranslation) {
                    return extractedTranslation;
                } else if (extractedError) {
                    throw new Error(extractedError);
                }
            }

            throw new Error('未获取到翻译结果');
        };

        // 修改：表单提交处理函数
        const submitForm = async (e) => {
            e.preventDefault();

            // 表单验证
            if (!text.value.trim()) {
                error.value = '输入文本不能为空';
                return;
            }

            if (text.value.length > 2000) {
                error.value = '输入文本过长，请限制在2000字符以内';
                return;
            }

            // 获取CSRF令牌
            const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

            try {
                // 分割文本为句子
                const sentences = splitTextIntoSentences(text.value);

                // 重置翻译结果和错误
                translatedText.value = '';
                error.value = '';

                // 显示总句数的信息
                showToast(`开始翻译 ${sentences.length} 个句子...`);

                // 逐句翻译
                const translatedSentences = [];
                for (let i = 0; i < sentences.length; i++) {
                    const sentence = sentences[i];

                    // 更新进度
                    progress.value = Math.round((i / sentences.length) * 100);
                    showToast(`正在翻译第 ${i + 1}/${sentences.length} 个句子 (${progress.value}%)...`);

                    // 翻译当前句子
                    try {
                        const translatedSentence = await translateSentence(sentence, mode.value, csrftoken);
                        translatedSentences.push(translatedSentence);
                    } catch (err) {
                        console.error(`翻译句子 "${sentence}" 时出错:`, err);
                        translatedSentences.push(`[翻译错误: ${err.message}]`);
                    }
                }

                // 智能合并所有翻译结果，避免多余空格
                translatedText.value = intelligentJoinSentences(translatedSentences, mode.value);

                // 完成提示
                showToast('翻译完成！');

            } catch (err) {
                console.error('翻译过程出错:', err);
                error.value = '翻译服务出现错误，请稍后重试';
            }
        };

        // 新增：智能合并句子的函数
        const intelligentJoinSentences = (sentences, mode) => {
            if (sentences.length === 0) return '';
            if (sentences.length === 1) return sentences[0];

            // 根据翻译模式决定合并策略
            if (mode === 'zh2en') {
                // 英文句子合并策略：检查标点和空格
                return sentences.reduce((result, sentence, index) => {
                    if (index === 0) return sentence.trim();

                    const prevSentence = result;
                    const currentSentence = sentence.trim();

                    // 检查前一个句子的结尾是否有标点符号
                    const endsWithPunctuation = /[.!?,:;]$/.test(prevSentence);

                    // 检查当前句子的开头是否有标点符号
                    const startsWithPunctuation = /^[.!?,:;]/.test(currentSentence);

                    if (endsWithPunctuation && !startsWithPunctuation) {
                        // 如果前一句以标点结尾，当前句不以标点开头，添加一个空格
                        return prevSentence + ' ' + currentSentence;
                    } else if (!endsWithPunctuation && !startsWithPunctuation) {
                        // 如果都没有标点，添加空格
                        return prevSentence + ' ' + currentSentence;
                    } else {
                        // 其他情况直接连接，避免多余空格
                        return prevSentence + currentSentence;
                    }
                }, '');
            } else {
                // 中文句子合并策略：大多数情况下不需要空格
                return sentences.reduce((result, sentence, index) => {
                    if (index === 0) return sentence.trim();

                    const prevSentence = result;
                    const currentSentence = sentence.trim();

                    // 检查当前句子是否以中文标点开头
                    const startsWithChinesePunctuation = /^[。，、；：？！""''（）【】《》]/.test(currentSentence);

                    // 对于中文，通常不需要空格，除非有特殊要求
                    if (startsWithChinesePunctuation) {
                        return prevSentence + currentSentence;
                    } else {
                        // 如果当前句不以标点开头，可能需要添加适当的标点
                        // 这里简单处理，直接连接
                        return prevSentence + currentSentence;
                    }
                }, '');
            }
        };

        // 显示提示框
        const showToast = (message) => {
            toastMessage.value = message;
            toastVisible.value = true;

            setTimeout(() => {
                toastVisible.value = false;
            }, 2000);
        };

        // 监听模式变化，清除结果
        watch(mode, (newMode, oldMode) => {
            if (newMode !== oldMode) {
                translatedText.value = '';
                error.value = '';
            }
        });

        return {
            text,
            translatedText,
            mode,
            error,
            toastMessage,
            toastVisible,
            progress,
            submitForm,
            showToast
        };
    },
    template: `
        <div class="container">
            <app-header></app-header>
            
            <form @submit="submitForm" id="translateForm">
                <language-switch v-model="mode"></language-switch>
                
                <div class="translation-container">
                    <input-area 
                        v-model="text" 
                        :mode="mode"
                        @show-toast="showToast">
                    </input-area>
                    
                    <translate-button></translate-button>
                    
                    <output-area 
                        :translated-text="translatedText" 
                        :error="error" 
                        :mode="mode"
                        @show-toast="showToast">
                    </output-area>
                </div>
            </form>
            
            <app-footer></app-footer>
            
            <toast 
                :message="toastMessage" 
                :is-visible="toastVisible"
                :progress="progress">
            </toast>
        </div>
    `
};

// 等待DOM加载完成后挂载Vue应用
document.addEventListener('DOMContentLoaded', () => {
    // 创建Vue应用并挂载
    const app = createApp(App);
    app.mount('#app');
}); 