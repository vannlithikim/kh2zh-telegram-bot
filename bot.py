import logging
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Optional, Tuple
from datetime import datetime
import os
from dotenv import load_dotenv

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()  # Load .env file
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("❌ Missing BOT_TOKEN in .env file!")

# NLLB Model Initialization
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # Faster on CPU

class NLLBTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.translation_count = 0
        self.last_translation_time = None
        
        # Language code mapping
        self.lang_codes = {
            'km': 'khm_Khmr',  # Khmer
            'zh': 'zho_Hans',  # Chinese (Simplified)
        }
        
        try:
            logger.info(f"Loading NLLB model ({MODEL_NAME}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)
            
            # Create language code to ID mapping
            self.lang_code_to_id = {
                'khm_Khmr': self.tokenizer.convert_tokens_to_ids('khm_Khmr'),
                'zho_Hans': self.tokenizer.convert_tokens_to_ids('zho_Hans')
            }
            
            self.model_loaded = True
            logger.info("✅ NLLB model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load NLLB model: {e}")

    def clean_text(self, text: str) -> str:
        """Clean input text by removing excessive whitespace and special characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Keep only Khmer, Chinese, and basic punctuation
        text = re.sub(r'[^\w\s\u1780-\u17FF\u4e00-\u9fff.,!?;:\'"()-]', '', text)
        return text

    async def translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.model_loaded:
            return None, "Model not loaded"
        
        text = self.clean_text(text)  # Now this will work
        if not text:
            return None, "Empty text"

        source_code = self.lang_codes.get(source_lang)
        target_code = self.lang_codes.get(target_lang)
        if not source_code or not target_code:
            return None, "Unsupported language"

        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.lang_code_to_id[target_code],
                max_length=512
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            self.translation_count += 1
            self.last_translation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return translated_text, "NLLB (Meta)"
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None, str(e)
translator = NLLBTranslator()

def detect_language(text):
    clean_text = re.sub(r'[^\w\u1780-\u17FF\u4e00-\u9fff]', '', text)
    if not clean_text:
        return None
    khmer_chars = sum(1 for char in clean_text if '\u1780' <= char <= '\u17FF')
    chinese_chars = sum(1 for char in clean_text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(clean_text)
    if total_chars == 0:
        return None
    khmer_ratio = khmer_chars / total_chars
    chinese_ratio = chinese_chars / total_chars
    if khmer_ratio > 0.1:
        return 'km'
    elif chinese_ratio > 0.1:
        return 'zh'
    return None

# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
    🌟 *Welcome to Khmer-Chinese Translator Bot!* 🌟

    I can translate between:
    - 🇰🇭 Khmer → 🇨🇳 Chinese
    - 🇨🇳 Chinese → 🇰🇭 Khmer

    Just send me text, and I'll translate it automatically!

    📌 *Commands:*
    /start - Show this welcome message
    /help - How to use the bot
    /status - Check bot system status
    """
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_msg = """
    ℹ️ *How to Use:*
    1. Send any Khmer or Chinese text.
    2. I'll detect the language and translate it automatically.

    🔍 *Examples:*
    - សួស្តី → 你好 (Khmer to Chinese)
    - 谢谢 → សូមអរគុណ (Chinese to Khmer)

    ⚠️ *Limitations:*
    - Max 1000 characters per message
    - Works best with clear text (no slang/rare words)
    """
    await update.message.reply_text(help_msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = f"""
    📊 *System Status*
    - Model: `{MODEL_NAME}`
    - Device: `{translator.device.upper()}`
    - Model Loaded: `{'✅' if translator.model_loaded else '❌'}`
    - Translations Completed: `{translator.translation_count}`
    - Last Translation: `{translator.last_translation_time or 'Never'}`
    """
    await update.message.reply_text(status_msg, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if not user_text:
        await update.message.reply_text("Please send some text to translate!")
        return
    if len(user_text) > 1000:
        await update.message.reply_text("Text too long! Keep it under 1000 characters.")
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    detected_lang = detect_language(user_text)
    if not detected_lang:
        await update.message.reply_text("Couldn't detect language. Use Khmer or Chinese.")
        return
    
    source_lang, target_lang = ('km', 'zh') if detected_lang == 'km' else ('zh', 'km')
    direction = "🇰🇭 Khmer → 🇨🇳 Chinese" if detected_lang == 'km' else "🇨🇳 Chinese → 🇰🇭 Khmer"
    
    try:
        translated_text, service_used = await translator.translate(user_text, source_lang, target_lang)
        if translated_text:
            response = f"🔄 *{direction}*\n\n📝 *Original:*\n{user_text}\n\n💬 *Translation:*\n{translated_text}"
            if service_used:
                response += f"\n\n_Powered by {service_used}_"
            await update.message.reply_text(response, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Translation failed. Try again later.")
    except Exception as e:
        logger.error(f"Translation error: {e}")
        await update.message.reply_text("❌ Error during translation.")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Command Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    
    # Message Handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start Bot
    logger.info("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()