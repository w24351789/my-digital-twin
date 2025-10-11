# 我的數位分身：個人化聊天機器人專案 (My Digital Twin)

本專案旨在打造一個高度個人化的 AI 聊天機器人，使其能夠學習並模仿我本人在 LINE 上的獨特聊天風格。透過在我真實的聊天記錄上微調 Qwen 語言模型，最終目標是創造出一個能夠以假亂真的「數位分身」。

## 核心功能 (Core Features)

* **風格模仿 (Style Mimicry)**: 精準複製我的常用詞彙（如 `Na`, `豪`, `窩`, `眠`）、表情符號、口頭禪及回應的簡潔性。
* **語氣還原 (Tone Replication)**: 學習我在不同情境下的語氣，包括日常閒聊、分享生活、與特定朋友的互動模式。
* **上下文感知 (Context Awareness)**: 能夠理解對話的上下文，並根據前幾輪的對話歷史，生成自然且連貫的回應。

## 技術選型 (Tech Stack)

* **基礎模型 (Base Model)**: `Qwen1.5-7B-Chat`
    * **選擇理由**: Qwen 系列模型對中文的支援度極佳，7B 的尺寸在消費級硬體上進行微調是可行的，且在性能和資源消耗之間取得了很好的平衡。
* **微調技術 (Fine-tuning Technique)**: PEFT (Parameter-Efficient Fine-tuning) with LoRA (Low-Rank Adaptation)
    * **選擇理由**: LoRA 讓我們無需訓練模型的全部參數，極大地降低了對顯示卡記憶體 (VRAM) 的需求，使得個人電腦上的微調成為可能。
* **主要框架與函式庫 (Frameworks & Libraries)**:
    * Python 3.9+
    * PyTorch
    * Hugging Face Transformers: 用於載入模型和 Tokenizer。
    * Hugging Face Datasets: 用於處理和載入資料集。
    * Hugging Face PEFT: 用於實現 LoRA。
    * Hugging Face TRL: 提供 `SFTTrainer` 簡化訓練流程。
    * `regex`: 用於解析 LINE 的 `.txt` 聊天記錄。

## 專案工作流程 (Workflow)

整個專案分為四個主要階段：資料處理、資料集建構、模型微調、以及互動推論。

### 1. 資料匯出與預處理

這是專案的基石，目標是將非結構化的 LINE 聊天記錄轉換為乾淨的結構化資料。

* **步驟**:
    1.  從 LINE App 中匯出指定對話的聊天記錄，得到一個 `.txt` 檔案。
    2.  執行一個自訂的 Python 腳本 (`parse_line.py`)，使用正則表達式解析 `.txt` 檔。
    3.  將每一行對話轉換為一個包含 `timestamp`, `sender`, `message` 的 JSON 物件。
    4.  **（重要）** 進行**隱私處理**：在此步驟中，自動化地移除或替換掉姓名、電話、地址、密碼等個人敏感資訊。

### 2. 資料集建構

將處理好的對話資料，轉換成模型微調時可以讀取的「對話格式」。我們將採用 ChatML 格式。

* **目標**: 將連續的對話轉換成多輪對話的訓練樣本。其中，**我 的發言**被標記為 `assistant` 角色，而**對方的發言**被標記為 `user` 角色。
* **輸出檔案**: `training_dataset.jsonl`
* **格式範例**:
    ```json
    // training_dataset.jsonl 的其中一行
    {
      "messages": [
        { "role": "user", "content": "我要到了" },
        { "role": "assistant", "content": "Na" }
      ]
    }
    ```
    *備註：一個 `messages` 列表可以包含多輪對話，以提供模型更豐富的上下文。*

### 3. 模型微調

使用處理好的資料集來訓練 Qwen 模型。

* **腳本**: `train.py`
* **流程**:
    1.  載入 `Qwen1.5-7B-Chat` 的基礎模型和 Tokenizer。
    2.  載入 `training_dataset.jsonl` 資料集。
    3.  設定 `LoraConfig`，指定要訓練的模組和 LoRA 參數。
    4.  初始化 `SFTTrainer`，傳入模型、資料集和 LoRA 配置。
    5.  開始訓練。訓練完成後，將 LoRA 適配器 (adapter) 權重儲存到指定目錄（例如 `./naomi-chatbot-adapter`）。

### 4. 互動推論

載入微調後的模型，並與你的「數位分身」進行即時對話。

* **腳本**: `chat.py`
* **流程**:
    1.  載入 `Qwen1.5-7B-Chat` 基礎模型。
    2.  從 `./naomi-chatbot-adapter` 目錄載入 LoRA 權重，並將其與基礎模型融合。
    3.  建立一個簡單的命令列迴圈，讓你可以輸入文字並看到模型的即時回應。
    4.  在推論時，同樣要將對話歷史格式化成 ChatML 格式傳遞給模型。

## 如何執行 (How to Run)

```bash
# 1. 安裝所需的 Python 函式庫
pip install torch transformers datasets peft trl

# 2. 處理原始聊天記錄
# 將你從 LINE 匯出的 'line_chat.txt' 放在 data/ 目錄下
python parse_line.py --input_file data/line_chat.txt --output_file data/training_dataset.jsonl --my_name "i"

# 3. 開始模型微調
python train.py

# 4. 與你的 AI 分身聊天
python chat.py