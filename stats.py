import asyncio
import json
import re
import aiosqlite
from datetime import datetime
from telethon import TelegramClient, events
from telethon.tl.types import UpdateBotMessageReaction
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import logging
from telethon.tl.types import InputBotInlineResult, InputBotInlineMessageText
from transformers import pipeline
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
print(np.__version__)

logging.basicConfig(level=logging.DEBUG)
# Define your API ID, API Hash, and bot token
api_id = '25965226'
api_hash = '7a1c735626be2bbb5b0898d66a47e15d'
bot_token = '7444376874:AAFg4BH-Kv5qQKFkPIhhGnEgJtsAM-sIv20'

client = TelegramClient('bot', api_id, api_hash)
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
db = None
sia = None
bot_user_id = None

async def init_bot():
    global bot_user_id
    bot = await client.get_me()
    bot_user_id = bot.id
    print(f"Bot user ID: {bot_user_id}")

async def init_db():
    global db, sia
    db = await aiosqlite.connect('stats.db')
    
    await db.execute('''
        CREATE TABLE IF NOT EXISTS engagement (
            user_id INTEGER,
            period TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, period)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS hourly_engagement (
            user_id INTEGER,
            hour INTEGER,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, hour)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS reactions (
            user_id INTEGER,
            reaction TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, reaction)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS participation (
            user_id INTEGER PRIMARY KEY,
            messages INTEGER DEFAULT 0,
            words INTEGER DEFAULT 0,
            characters INTEGER DEFAULT 0,
            replies INTEGER DEFAULT 0
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_lengths (
            user_id INTEGER PRIMARY KEY,
            lengths TEXT
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS media_usage (
            user_id INTEGER,
            media_type TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, media_type)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS emojis (
            emoji TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            user_id INTEGER,
            message_id INTEGER PRIMARY KEY,
            sentiment REAL
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_edits (
            user_id INTEGER PRIMARY KEY,
            edit_count INTEGER DEFAULT 0
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            user_id INTEGER,
            command TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, command)
        )
    ''')

    # Create the voice_message_lengths table if it does not exist
    await db.execute('''
        CREATE TABLE IF NOT EXISTS voice_message_lengths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            length INTEGER
        )
    ''')
    
    # Create the photo_sizes table if it does not exist
    await db.execute('''
        CREATE TABLE IF NOT EXISTS photo_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            size INTEGER,
            width INTEGER,
            height INTEGER
        )
    ''')

    # Create the video_sizes table if it does not exist
    await db.execute('''
        CREATE TABLE IF NOT EXISTS video_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            size INTEGER,
            duration INTEGER
        )
    ''')

    # Create the mentions table if it does not exist
    await db.execute('''
        CREATE TABLE IF NOT EXISTS mentions (
            user_id INTEGER,
            mention TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, mention)
        )
    ''')


    await db.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    await db.commit()

    # Initialize sentiment analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

async def update_engagement(user_id, period):
    await db.execute('''
        INSERT INTO engagement (user_id, period, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, period) DO UPDATE SET
        count = count + 1
    ''', (user_id, period))
    await db.commit()

async def update_hourly_engagement(user_id, hour):
    await db.execute('''
        INSERT INTO hourly_engagement (user_id, hour, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, hour) DO UPDATE SET
        count = count + 1
    ''', (user_id, hour))
    await db.commit()

async def update_reaction(user_id, reaction):
    await db.execute('''
        INSERT INTO reactions (user_id, reaction, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, reaction) DO UPDATE SET
        count = count + 1
    ''', (user_id, reaction))
    await db.commit()

async def update_participation(user_id, messages, words, characters, replies):
    await db.execute('''
        INSERT INTO participation (user_id, messages, words, characters, replies)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
        messages = participation.messages + excluded.messages,
        words = participation.words + excluded.words,
        characters = participation.characters + excluded.characters,
        replies = participation.replies + excluded.replies
    ''', (user_id, messages, words, characters, replies))
    await db.commit()

async def update_message_lengths(user_id, length):
    async with db.execute('SELECT lengths FROM message_lengths WHERE user_id = ?', (user_id,)) as cursor:
        current_lengths = await cursor.fetchone()
    if current_lengths:
        lengths = json.loads(current_lengths[0])
        lengths.append(length)
    else:
        lengths = [length]
    await db.execute('''
        INSERT INTO message_lengths (user_id, lengths)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
        lengths = ?
    ''', (user_id, json.dumps(lengths), json.dumps(lengths)))
    await db.commit()

async def update_media_usage(user_id, media_type):
    await db.execute('''
        INSERT INTO media_usage (user_id, media_type, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, media_type) DO UPDATE SET
        count = count + 1
    ''', (user_id, media_type))
    await db.commit()

async def update_voice_message_length(user_id, length):
    await db.execute('''
        INSERT INTO voice_message_lengths (user_id, length)
        VALUES (?, ?)
    ''', (user_id, length))
    await db.commit()
    logging.debug(f"Inserted voice message length: user_id={user_id}, length={length}")

async def update_photo_size(user_id, size, width, height):
    await db.execute('''
        INSERT INTO photo_sizes (user_id, size, width, height)
        VALUES (?, ?, ?, ?)
    ''', (user_id, size, width, height))
    await db.commit()
    logging.debug(f"Inserted photo size: user_id={user_id}, size={size}, width={width}, height={height}")

async def update_video_size(user_id, size, duration):
    await db.execute('''
        INSERT INTO video_sizes (user_id, size, duration)
        VALUES (?, ?, ?)
    ''', (user_id, size, duration))
    await db.commit()
    logging.debug(f"Inserted video size: user_id={user_id}, size={size}, duration={duration}")


async def save_message_sentiment(user_id, message_id, sentiment_score):
    logging.debug(f"Saving sentiment: user_id={user_id}, message_id={message_id}, sentiment_score={sentiment_score}")
    
    await db.execute('''
        INSERT OR IGNORE INTO sentiments (user_id, message_id, sentiment)
        VALUES (?, ?, ?)
    ''', (user_id, message_id, sentiment_score))

    await db.execute('''
        UPDATE sentiments
        SET sentiment = ?
        WHERE message_id = ?
    ''', (sentiment_score, message_id))

    await db.commit()
    logging.debug(f"Sentiment saved: user_id={user_id}, message_id={message_id}, sentiment_score={sentiment_score}")


async def update_message_edits(user_id):
    await db.execute('''
        INSERT INTO message_edits (user_id, edit_count)
        VALUES (?, 1)
        ON CONFLICT(user_id) DO UPDATE SET
        edit_count = message_edits.edit_count + 1
    ''', (user_id,))
    await db.commit()

async def update_commands(user_id, command):
    await db.execute('''
        INSERT INTO commands (user_id, command, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, command) DO UPDATE SET
        count = count + 1
    ''', (user_id, command))
    await db.commit()

async def update_mention(user_id, mention):
    await db.execute('''
        INSERT INTO mentions (user_id, mention, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, mention) DO UPDATE SET
        count = count + 1
    ''', (user_id, mention))
    await db.commit()

async def update_emoji(emoji):
    await db.execute('''
        INSERT INTO emojis (emoji, count)
        VALUES (?, 1)
        ON CONFLICT(emoji) DO UPDATE SET
        count = count + 1
    ''', (emoji,))
    await db.commit()




@client.on(events.InlineQuery)
async def inline_query_handler(event):
    query = event.text
    if query.isdigit():
        user_id = int(query)
        results = await asyncio.gather(
            get_user_participation(user_id),
            get_user_hourly_engagement(user_id),
            get_user_reactions(user_id),
            get_user_media_usage(user_id),
            get_user_message_lengths(user_id),
            get_user_message_edits(user_id),
            get_user_commands(user_id),
            get_user_mentions(user_id),
            get_user_sentiment_analysis(user_id)
        )

        (participation_stats, hourly_engagement, reactions, media_usage, 
         message_lengths, message_edits, commands, mentions, average_sentiment) = results

        if average_sentiment is None:
            average_sentiment = 0

        user_stats = f"""
📊 **User Stats for [User {user_id}](tg://user?id={user_id})** 📊

📈 **Participation:**
{participation_stats}

⏰ **Hourly Engagement:**
{hourly_engagement}

🎉 **Reactions:**
{reactions}

🎥 **Media Usage:**
{media_usage}

📝 **Message Lengths:**
{message_lengths}

✏️ **Message Edits:**
{message_edits}

🔧 **Commands:**
{commands}

🔍 **Mentions:**
{mentions}

📊 **Average Sentiment:**
{average_sentiment:.2f}%
"""
        result = [
            InputBotInlineResult(
                id='1',
                type='article',
                title=f"Stats for User {user_id}",
                description="View the detailed stats",
                send_message=InputBotInlineMessageText(user_stats)
            )
        ]

        await event.answer(result)
    else:
        await event.answer([], switch_pm='Enter a valid user ID', switch_pm_param='start')


@client.on(events.NewMessage(pattern='/search'))
async def search_command_handler(event):
    try:
        query = event.message.message.split()
        if len(query) < 2:
            await event.reply("Please provide a user ID to search for. Usage: /search <user_id>")
            return
        
        user_id = int(query[1])
        results = await asyncio.gather(
            get_user_participation(user_id),
            get_user_hourly_engagement(user_id),
            get_user_reactions(user_id),
            get_user_media_usage(user_id),
            get_user_message_lengths(user_id),
            get_user_message_edits(user_id),
            get_user_commands(user_id),
            get_user_mentions(user_id),
            get_user_sentiment_analysis(user_id)
        )

        (participation_stats, hourly_engagement, reactions, media_usage, 
         message_lengths, message_edits, commands, mentions, average_sentiment) = results

        if average_sentiment is None:
            average_sentiment = 0

        user_stats = f"""
📊 **User Stats for [User {user_id}](tg://user?id={user_id})** 📊

📈 **Participation:**
{participation_stats}

⏰ **Hourly Engagement:**
{hourly_engagement}

🎉 **Reactions:**
{reactions}

🎥 **Media Usage:**
{media_usage}

📝 **Message Lengths:**
{message_lengths}

✏️ **Message Edits:**
{message_edits}

🔧 **Commands:**
{commands}

🔍 **Mentions:**
{mentions}

📊 **Average Sentiment:**
{average_sentiment:.2f}%
"""

        await event.reply(user_stats)
    except ValueError:
        await event.reply("Invalid user ID. Please provide a numeric user ID.")
    except Exception as e:
        await event.reply(f"An error occurred: {e}")

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

mention_pattern = re.compile(r'@\w+')


@client.on(events.NewMessage(incoming=True))
async def track_engagement(event):
    global bot_user_id
    user_id = event.sender_id
    if user_id == bot_user_id:
        return  # Skip the bot's own messages

    message_id = event.message.id
    hour = datetime.now().hour
    message_text = event.message.message or ""

    async def handle_message():
        await asyncio.gather(
            update_engagement(user_id, 'daily'),
            update_engagement(user_id, 'weekly'),
            update_engagement(user_id, 'monthly'),
            update_hourly_engagement(user_id, hour)
        )
        await process_message_content(event, message_text, user_id)
        await process_media_types(event, user_id)

        # Insert the message into the messages table
        await db.execute('''
            INSERT INTO messages (user_id, message, timestamp)
            VALUES (?, ?, ?)
        ''', (user_id, message_text, datetime.now()))
        await db.commit()

        # Sentiment analysis
        stars = await analyze_sentiment(message_text)
        sentiment_score = (stars / 5) * 100  # Convert to percentage
        await save_message_sentiment(user_id, message_id, sentiment_score)

        # Track replies, forwarded messages, links, commands, and mentions
        if event.message.is_reply:
            reply_to_user_id = (await event.get_reply_message()).sender_id
            await update_participation(reply_to_user_id, 0, 0, 0, 1)
        if event.message.fwd_from:
            await update_engagement(user_id, 'forwarded')
        if "http://" in message_text or "https://" in message_text:
            await update_media_usage(user_id, 'link')
        if message_text.startswith('/'):
            command = message_text.split()[0]
            await update_commands(user_id, command)
        mentions = mention_pattern.findall(message_text)
        for mention in mentions:
            await update_mention(user_id, mention)

    asyncio.create_task(handle_message())

@client.on(events.MessageEdited)
async def track_edits(event):
    user_id = event.sender_id

    async def handle_edit():
        await update_message_edits(user_id)

    asyncio.create_task(handle_edit())

async def process_message_content(event, message_text, user_id):
    if message_text:
        words = len(message_text.split())
        characters = len(message_text)
        await asyncio.gather(
            update_participation(user_id, 1, words, characters, 0),
            update_message_lengths(user_id, characters)
        )

        # Find all individual emojis in the message text
        emojis = emoji_pattern.findall(message_text)
        for emoji in emojis:
            for individual_emoji in list(emoji):
                await update_emoji(individual_emoji)

    if event.message.is_reply:
        await update_participation(user_id, 0, 0, 0, 1)

async def process_media_types(event, user_id):
    if event.message.gif:
        await update_media_usage(user_id, 'gif')
    elif event.message.video:
        await update_media_usage(user_id, 'video')
        file_size = event.message.file.size if event.message.file else 0
        duration = event.message.video.attributes[0].duration
        await update_video_size(user_id, file_size, duration)
    elif event.message.sticker:
        await update_media_usage(user_id, 'sticker')
    elif event.message.voice:
        await update_media_usage(user_id, 'voice')
        voice_length = event.message.voice.attributes[0].duration
        await update_voice_message_length(user_id, voice_length)
    elif isinstance(event.message.media, MessageMediaPhoto):
        await update_media_usage(user_id, 'photo')
        file_size = event.message.file.size if event.message.file else 0
        largest_photo = max(event.message.photo.sizes, key=lambda s: s.w * s.h)
        width = largest_photo.w
        height = largest_photo.h
        await update_photo_size(user_id, file_size, width, height)

@client.on(events.Raw)
async def raw_update_handler(event):
    if isinstance(event, UpdateBotMessageReaction):
        user_id = event.actor.user_id
        if user_id == bot_user_id:
            return  # Skip the bot's own reactions

        new_reactions = event.new_reactions

        async def handle_reaction():
            if new_reactions:
                for reaction in new_reactions:
                    reaction_str = reaction.emoticon
                    await update_reaction(user_id, reaction_str)

        asyncio.create_task(handle_reaction())

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    welcome_message = (
        "👋 Welcome to the Stats Bot!\n\n"
        "Use the following commands to interact with the bot:\n"
        "/stats - Get the latest group statistics\n"
        "/help - Get help on how to use the bot\n\n"
        "Enjoy using the bot!"
    )
    await event.respond(welcome_message)

@client.on(events.NewMessage(pattern='/stats'))
async def send_combined_stats(event):
    try:
        # Get all the results
        results = await asyncio.gather(
            generate_text_summary(),
            get_most_active_users(),
            get_engagement_by_time_of_day(),
            get_media_type_distribution(),
            calculate_average_edits(),
            calculate_average_voice_length(),
            calculate_average_photo_size(),
            calculate_average_video_size_and_duration(),
            get_general_sentiment(),
            prepare_data(),
            generate_sentiment_report()  # Include sentiment report
        )

        # Extract individual results
        text_summary = results[0]
        most_active_users = results[1]
        engagement_by_time = results[2]
        media_type_distribution = results[3]
        avg_edits = results[4]
        avg_voice_length = results[5]
        photo_stats = results[6]
        video_stats = results[7]
        general_sentiment = results[8]
        data = results[9]
        sentiment_report = results[10]

        # Predict future activity if there's enough data
        if data.shape[0] >= 2:
            forecast = await train_and_predict(data)
            forecast_plot = await generate_forecast_plot(forecast)
            forecast_caption = "📊 Group Activity Forecast for the Next 30 Days 📊"
        else:
            forecast_plot = None
            forecast_caption = "Not enough data to make a prediction."

        if general_sentiment is None:
            general_sentiment = 0

        # Combine text results
        combined_stats = "\n\n".join(map(str, [
            text_summary,
            most_active_users,
            engagement_by_time,
            media_type_distribution,
            f"✏️ **Average Message Edits per User**: {avg_edits:.2f}",
            f"🎤 **Average Voice Message Length**: {avg_voice_length:.2f} seconds",
            f"📸 **Average Photo Size**: {photo_stats['avg_size_mb']:.2f} MB",
            f"📐 **Average Photo Dimensions**: {photo_stats['avg_width']:.2f} x {photo_stats['avg_height']:.2f} pixels",
            f"🎥 **Average Video Size**: {video_stats['avg_size_mb']:.2f} MB",
            f"⏳ **Average Video Duration**: {video_stats['avg_duration']:.2f} seconds",
            f"🌐 **General Sentiment of All Users**: {general_sentiment:.2f}%",
            sentiment_report  # Include sentiment report
        ]))

        # Send the combined text stats
        if len(combined_stats) > 4096:
            parts = [combined_stats[i:i + 4096] for i in range(0, len(combined_stats), 4096)]
            for part in parts:
                await client.send_message(event.chat_id, part)
        else:
            await client.send_message(event.chat_id, combined_stats)

        # Send the forecast plot if available
        if forecast_plot:
            await client.send_file(event.chat_id, forecast_plot, caption=forecast_caption)
        else:
            return

    except Exception as e:
        await event.reply(f"Error: {e}")

    except Exception as e:
        await event.reply(f"Error: {e}")

async def generate_text_summary():
    results = await asyncio.gather(
        format_engagement_leaderboard('daily'),
        format_engagement_leaderboard('weekly'),
        format_engagement_leaderboard('monthly'),
        format_active_hours(),
        format_reaction_leaderboard(),
        format_popular_reactions(),
        format_participation_stats(),
        format_message_length_stats(),
        format_media_stats('gif'),
        format_media_stats('video'),
        format_media_stats('sticker'),
        format_emoji_stats()
    )

    (daily_leaderboard, weekly_leaderboard, monthly_leaderboard, active_hours, 
     reaction_leaderboard, popular_reactions, participation_stats, 
     message_length_stats, gif_stats, video_stats, sticker_stats, 
     emoji_stats) = results

    message = (
        "📊 **Group Stats** 📊\n\n"
        f"{daily_leaderboard}\n"
        f"{weekly_leaderboard}\n"
        f"{monthly_leaderboard}\n"
        f"{active_hours}\n"
        f"{reaction_leaderboard}\n"
        f"{popular_reactions}\n"
        f"{participation_stats}\n"
        f"{message_length_stats}\n"
        f"{gif_stats}\n"
        f"{video_stats}\n"
        f"{sticker_stats}\n"
        f"{emoji_stats}\n"
    )

    return message


async def get_user_sentiment_scores():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('''
            SELECT user_id, AVG(sentiment) as avg_sentiment
            FROM sentiments
            GROUP BY user_id
        ''') as cursor:
            rows = await cursor.fetchall()
    return rows


async def get_extreme_sentiment_users():
    sentiment_scores = await get_user_sentiment_scores()
    
    if not sentiment_scores:
        return None, None

    # Sort users by sentiment score
    sentiment_scores.sort(key=lambda x: x[1])

    most_negative_user = sentiment_scores[0]
    most_positive_user = sentiment_scores[-1]

    return most_negative_user, most_positive_user



async def generate_sentiment_report():
    most_negative_user, most_positive_user = await get_extreme_sentiment_users()

    if most_negative_user and most_positive_user:
        report = f"""
📊 **User Sentiment Report** 📊

🔴 **Most Negative User**:
- User ID: {most_negative_user[0]}
- Average Sentiment Score: {most_negative_user[1]:.2f}

🟢 **Most Positive User**:
- User ID: {most_positive_user[0]}
- Average Sentiment Score: {most_positive_user[1]:.2f}
"""
    else:
        report = "No sentiment data available."

    return report


async def get_general_sentiment():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT sentiment FROM sentiments') as cursor:
            rows = await cursor.fetchall()
    
    if rows:
        sentiments = [row[0] for row in rows]
        general_sentiment = sum(sentiments) / len(sentiments)
        return general_sentiment
    return None

async def analyze_sentiment(message_text):
    result = sentiment_analyzer(message_text)
    # The model returns a label '1 star' to '5 stars'
    label = result[0]['label']
    stars = int(label[0])  # Extract the number of stars from the label
    return stars

async def calculate_average_edits():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT AVG(edit_count) FROM message_edits') as cursor:
            average_edits = await cursor.fetchone()
    logging.debug(f"Average message edits calculation result: {average_edits}")
    return average_edits[0] if average_edits and average_edits[0] is not None else 0

async def get_most_active_users():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, messages FROM participation ORDER BY messages DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "📈 **Top 10 Most Active Users** 📈\n"
    for rank, (user_id, message_count) in enumerate(rows, start=1):
        result += f"{rank}. [User {user_id}](tg://user?id={user_id}) - {message_count} messages\n"
    return result

async def get_engagement_by_time_of_day():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT hour, SUM(count) as message_count FROM hourly_engagement GROUP BY hour ORDER BY message_count DESC') as cursor:
            rows = await cursor.fetchall()
    result = "⏰ **Engagement by Time of Day** ⏰\n"
    for hour, message_count in rows:
        result += f"{hour}:00 - {message_count} messages\n"
    return result

async def get_media_type_distribution():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT media_type, SUM(count) as usage_count FROM media_usage GROUP BY media_type ORDER BY usage_count DESC') as cursor:
            rows = await cursor.fetchall()
    result = "🎥 **Media Type Distribution** 🎥\n"
    for media_type, usage_count in rows:
        result += f"{media_type.capitalize()}: {usage_count} uses\n"
    return result

async def format_engagement_leaderboard(period):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, count FROM engagement WHERE period = ? ORDER BY count DESC LIMIT 10', (period,)) as cursor:
            rows = await cursor.fetchall()
    result = f"🏆 **Top 10 Active Users ({period.capitalize()})** 🏆\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        result += f"{rank}. [User {user_id}](tg://user?id={user_id}) - {count} messages\n"
    return result

async def format_active_hours():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT hour, SUM(count) FROM hourly_engagement GROUP BY hour ORDER BY SUM(count) DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "⏰ **Top 10 Active Hours** ⏰\n"
    for rank, (hour, count) in enumerate(rows, start=1):
        result += f"{rank}. {hour}:00 - {count} messages\n"
    return result

async def format_reaction_leaderboard():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, SUM(count) FROM reactions GROUP BY user_id ORDER BY SUM(count) DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "🏆 **Top 10 Users by Reactions Received** 🏆\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        result += f"{rank}. [User {user_id}](tg://user?id={user_id}) - {count} reactions\n"
    return result

async def format_popular_reactions():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT reaction, SUM(count) FROM reactions GROUP BY reaction ORDER BY SUM(count) DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "🎉 **Top 10 Popular Reactions** 🎉\n"
    for rank, (reaction, count) in enumerate(rows, start=1):
        result += f"{rank}. {reaction} - {count} uses\n"
    return result

async def format_participation_stats():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, messages, words, characters, replies FROM participation ORDER BY messages DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "📈 **Top 10 Participation Stats** 📈\n"
    for rank, (user_id, messages, words, characters, replies) in enumerate(rows, start=1):
        result += (f"{rank}. [User {user_id}](tg://user?id={user_id}) - "
                   f"{messages} messages, "
                   f"{words} words, "
                   f"{characters} characters, "
                   f"{replies} replies\n")
    return result

async def format_message_length_stats():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, lengths FROM message_lengths') as cursor:
            rows = await cursor.fetchall()
    sorted_data = sorted(
        ((user_id, sum(json.loads(lengths)) / len(json.loads(lengths)) if json.loads(lengths) else 0) for user_id, lengths in rows),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    result = "📝 **Top 10 Average Message Lengths** 📝\n"
    for rank, (user_id, avg_length) in enumerate(sorted_data, start=1):
        result += f"{rank}. [User {user_id}](tg://user?id={user_id}) - {avg_length:.2f} characters/message\n"
    return result

async def format_media_stats(media_type):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT user_id, count FROM media_usage WHERE media_type = ? ORDER BY count DESC LIMIT 10', (media_type,)) as cursor:
            rows = await cursor.fetchall()
    result = f"**{media_type.capitalize()} Stats (Top 10 Users):**\n"
    for user_id, count in rows:
        result += f" - {user_id}: {count} {media_type}s\n"
    return result

async def format_emoji_stats():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT emoji, count FROM emojis ORDER BY count DESC LIMIT 10') as cursor:
            rows = await cursor.fetchall()
    result = "**Emoji Stats (Top 10 Emojis):**\n"
    for emoji, count in rows:
        result += f" - {emoji}: {count} times\n"
    return result

async def calculate_average_voice_length():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT AVG(length) FROM voice_message_lengths') as cursor:
            result = await cursor.fetchone()
    logging.debug(f"Average voice length calculation result: {result}")
    return result[0] if result and result[0] is not None else 0

async def calculate_average_photo_size():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT AVG(size) / 1024.0 / 1024.0 AS avg_size_mb, AVG(width) AS avg_width, AVG(height) AS avg_height FROM photo_sizes') as cursor:
            result = await cursor.fetchone()
    logging.debug(f"Average photo size calculation result: {result}")
    return {
        "avg_size_mb": result[0] if result and result[0] is not None else 0,
        "avg_width": result[1] if result and result[1] is not None else 0,
        "avg_height": result[2] if result and result[2] is not None else 0
    }

async def calculate_average_video_size_and_duration():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT AVG(size) / 1024.0 / 1024.0 AS avg_size_mb, AVG(duration) AS avg_duration FROM video_sizes') as cursor:
            result = await cursor.fetchone()
    logging.debug(f"Average video size and duration calculation result: {result}")
    return {
        "avg_size_mb": result[0] if result and result[0] is not None else 0,
        "avg_duration": result[1] if result and result[1] is not None else 0

    }


async def get_user_participation(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT messages, words, characters, replies FROM participation WHERE user_id = ?', (user_id,)) as cursor:
            row = await cursor.fetchone()
    if row:
        return f"Messages: {row[0]}\nWords: {row[1]}\nCharacters: {row[2]}\nReplies: {row[3]}"
    return "No participation data."

async def get_user_hourly_engagement(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT hour, count FROM hourly_engagement WHERE user_id = ? ORDER BY hour', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    result = ""
    for hour, count in rows:
        result += f"{hour}:00 - {count} messages\n"
    return result if result else "No hourly engagement data."

async def get_user_reactions(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT reaction, count FROM reactions WHERE user_id = ? ORDER BY count DESC', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    result = ""
    for reaction, count in rows:
        result += f"{reaction} - {count} times\n"
    return result if result else "No reaction data."

async def get_user_media_usage(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT media_type, count FROM media_usage WHERE user_id = ? ORDER BY count DESC', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    result = ""
    for media_type, count in rows:
        result += f"{media_type.capitalize()}: {count} uses\n"
    return result if result else "No media usage data."

async def get_user_sentiment_analysis(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT sentiment FROM sentiments WHERE user_id = ?', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    
    if rows:
        sentiments = [row[0] for row in rows]
        average_sentiment = sum(sentiments) / len(sentiments)
        return average_sentiment
    return None

async def get_user_message_lengths(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT lengths FROM message_lengths WHERE user_id = ?', (user_id,)) as cursor:
            row = await cursor.fetchone()
    if row:
        lengths = json.loads(row[0])
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        return f"Average Length: {avg_length:.2f} characters"
    return "No message length data."

async def get_user_message_edits(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT edit_count FROM message_edits WHERE user_id = ?', (user_id,)) as cursor:
            row = await cursor.fetchone()
    return f"Message Edits: {row[0]}" if row else "No message edits data."

async def get_user_commands(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT command, count FROM commands WHERE user_id = ? ORDER BY count DESC', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    result = ""
    for command, count in rows:
        result += f"{command}: {count} times\n"
    return result if result else "No commands data."

async def get_user_mentions(user_id):
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT mention, count FROM mentions WHERE user_id = ? ORDER BY count DESC', (user_id,)) as cursor:
            rows = await cursor.fetchall()
    result = ""
    for mention, count in rows:
        result += f"{mention}: {count} times\n"
    return result if result else "No mentions data."




async def fetch_daily_message_counts():
    async with aiosqlite.connect('stats.db') as db:
        async with db.execute('SELECT date(timestamp) as date, COUNT(*) as message_count FROM messages GROUP BY date ORDER BY date') as cursor:
            rows = await cursor.fetchall()
    return rows

async def prepare_data():
    rows = await fetch_daily_message_counts()
    data = pd.DataFrame(rows, columns=['ds', 'y'])  # Prophet requires columns named 'ds' and 'y'
    return data

async def train_and_predict(data):
    if data.shape[0] < 2:
        raise ValueError("Not enough data to make a prediction. At least 2 data points are required.")
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)  # Predict for the next 30 days
    forecast = model.predict(future)
    return forecast

async def generate_forecast_plot(forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(forecast, ax=ax)
    plt.title('Group Activity Forecast')
    plt.xlabel('Date')
    plt.ylabel('Messages Sent')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf






async def main():
    await init_db()
    await client.start(bot_token=bot_token)
    if not await client.is_user_authorized():
        raise ConnectionError("Bot is not authorized. Check your bot token.")
    await init_bot()  # Initialize bot and get bot user ID
    await client.run_until_disconnected()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
