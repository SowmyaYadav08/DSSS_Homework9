from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"C:\Users\sowmy\PycharmProjects\DSSS_Homework_9\tinyllama"

# Loading the model and tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # we are using eos_token in pad_token to avoid additional embeddings,
    # simplifying token handling and End of sequence behaviour in decoding
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Function to generate a response using TinyLlama
def generate_response(prompt: str) -> str:
    # Prepend context to guide the model
    formatted_prompt = (
        "You are a helpful assistant. Provide detailed and useful responses to user queries.\n\n"
        f"User: {prompt}\nAssistant:"
    )

    # Tokenize the input with padding
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)

    # Using attention mask to inform the model about padding
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # To generate the response
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=500,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    # To decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt context from the response (if present)
    response = response.replace(formatted_prompt, "").strip()
    return response

# Method to handle sending messages ( sends a message to the user)
async def send_message(update: Update, text: str) -> None:
    await update.message.reply_text(text)

# Method to handle receiving and processing user messages (receives and process a user message)
async def receive_message(update: Update) -> str:
    user_message = update.message.text
    response = generate_response(user_message)
    return response

# Telegram bot command to handle /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_message(update, "Hello! Ask me anything!")

# Telegram bot handler for messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response = await receive_message(update)
    await send_message(update, response)

def main():

    TELEGRAM_API_TOKEN = "7695363341:AAHEdaBDV0g2TbzJTjqfpO5xARML9eIhssc"

    # Initialize the Telegram bot application
    application = Application.builder().token(TELEGRAM_API_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
