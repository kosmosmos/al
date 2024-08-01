import asyncio
import aiosqlite
from time import time
import json
import re
from datetime import datetime
from telethon import TelegramClient, events
from telethon.tl.types import UpdateBotMessageReaction, MessageMediaPhoto
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from telethon.tl.types import InputBotInlineResult, InputBotInlineMessageText,ChannelParticipantsAdmins, ChannelParticipantCreator
from transformers import pipeline
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
from telethon import Button
from telethon.tl.types import PeerChannel
from telethon.tl.types import (
    PeerUser,
    MessageService,
    MessageActionChatAddUser,
    MessageActionChatJoinedByLink,
    MessageActionChatDeleteUser,
    UpdateNewChannelMessage,
    UpdateChannelParticipant,
    DocumentAttributeVideo
)

logging.basicConfig(level=logging.DEBUG)
api_id = '25965226'
api_hash = '7a1c735626be2bbb5b0898d66a47e15d'
bot_token = '7444376874:AAFg4BH-Kv5qQKFkPIhhGnEgJtsAM-sIv20'

client = TelegramClient('bot', api_id, api_hash)
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
db = None
sia = None
bot_user_id = None
user_cache = {}

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
            group_id INTEGER,
            period TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, period)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS hourly_engagement (
            user_id INTEGER,
            group_id INTEGER,
            hour INTEGER,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, hour)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS reactions (
            user_id INTEGER,
            group_id INTEGER,
            reaction TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, reaction)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS participation (
            user_id INTEGER,
            group_id INTEGER,
            messages INTEGER DEFAULT 0,
            words INTEGER DEFAULT 0,
            characters INTEGER DEFAULT 0,
            replies INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_lengths (
            user_id INTEGER,
            group_id INTEGER,
            lengths TEXT,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS media_usage (
            user_id INTEGER,
            group_id INTEGER,
            media_type TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, media_type)
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
            group_id INTEGER,
            message_id INTEGER,
            sentiment REAL,
            PRIMARY KEY (user_id, group_id, message_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_edits (
            user_id INTEGER,
            group_id INTEGER,
            edit_count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            user_id INTEGER,
            group_id INTEGER,
            command TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, command)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS voice_message_lengths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            length INTEGER
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS photo_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            size INTEGER,
            width INTEGER,
            height INTEGER
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS video_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            size INTEGER,
            duration INTEGER
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS mentions (
            user_id INTEGER,
            group_id INTEGER,
            mention TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, mention)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS animated_stickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            count INTEGER DEFAULT 0
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            length INTEGER,
            size INTEGER
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS forwards (
            user_id INTEGER,
            group_id INTEGER,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS link_sharing (
            user_id INTEGER,
            group_id INTEGER,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_deletions (
            user_id INTEGER,
            group_id INTEGER,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS message_types (
            user_id INTEGER,
            group_id INTEGER,
            message_type TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, group_id, message_type)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            link TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS group_admins (
            group_id INTEGER,
            user_id INTEGER,
            is_owner BOOLEAN,
            PRIMARY KEY (group_id, user_id)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS user_joins (
            user_id INTEGER,
            group_id INTEGER,
            join_method TEXT,
            join_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, group_id, join_date)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS user_leaves (
            user_id INTEGER,
            group_id INTEGER,
            leave_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, group_id, leave_date)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS user_additions (
            added_user_id INTEGER,
            group_id INTEGER,
            added_by_user_id INTEGER,
            add_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (added_user_id, group_id, add_date)
        )
    ''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS join_add_summary (
            group_id INTEGER PRIMARY KEY,
            total_joins INTEGER DEFAULT 0,
            joins_by_link INTEGER DEFAULT 0,
            total_additions INTEGER DEFAULT 0
        )
    ''')
    await db.execute('''
    CREATE TABLE IF NOT EXISTS video_selfies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        group_id INTEGER,
        size INTEGER,
        duration INTEGER
    )
''')
    await db.execute('''
        CREATE TABLE IF NOT EXISTS video_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            group_id INTEGER,
            size INTEGER,
            duration INTEGER,
            is_selfie INTEGER DEFAULT 0
        )
    ''')




    await db.execute('CREATE INDEX IF NOT EXISTS idx_engagement_user_group_period ON engagement (user_id, group_id, period)')
    await db.execute('CREATE INDEX IF NOT EXISTS idx_hourly_engagement_user_group_hour ON hourly_engagement (user_id, group_id, hour)')
    await db.commit()

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()


async def batch_update_participation(updates):
    async with db.execute('BEGIN TRANSACTION'):
        for update in updates:
            await db.execute('''
                INSERT INTO participation (user_id, group_id, messages, words, characters, replies)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, group_id) DO UPDATE SET
                messages = participation.messages + excluded.messages,
                words = participation.words + excluded.words,
                characters = participation.characters + excluded.characters,
                replies = participation.replies + excluded.replies
            ''', update)
    await db.commit()

async def is_admin(user_id, chat_id):
    async for user in client.iter_participants(chat_id, filter=ChannelParticipantsAdmins):
        if user.id == user_id:
            return True
    return False



async def get_user_entity(user_id):
    if user_id not in user_cache:
        user_cache[user_id] = await client.get_entity(user_id)
    return user_cache[user_id]



async def is_admin_in_db(chat_id, user_id):
    async with db.execute('SELECT 1 FROM group_admins WHERE group_id = ? AND user_id = ?', (chat_id, user_id)) as cursor:
        return await cursor.fetchone() is not None


async def update_user_joins(user_id, group_id, join_method):
    await db.execute('''
        INSERT INTO user_joins (user_id, group_id, join_method)
        VALUES (?, ?, ?)
    ''', (user_id, group_id, join_method))
    
    if join_method == 'link':
        await db.execute('''
            INSERT INTO join_add_summary (group_id, total_joins, joins_by_link)
            VALUES (?, 1, 1)
            ON CONFLICT(group_id) DO UPDATE SET
            total_joins = join_add_summary.total_joins + 1,
            joins_by_link = join_add_summary.joins_by_link + 1
        ''', (group_id,))
    else:
        await db.execute('''
            INSERT INTO join_add_summary (group_id, total_joins)
            VALUES (?, 1)
            ON CONFLICT(group_id) DO UPDATE SET
            total_joins = join_add_summary.total_joins + 1
        ''', (group_id,))

    await db.commit()

async def update_user_additions(added_user_id, group_id, added_by_user_id):
    await db.execute('''
        INSERT INTO user_additions (added_user_id, group_id, added_by_user_id)
        VALUES (?, ?, ?)
    ''', (added_user_id, group_id, added_by_user_id))
    
    await db.execute('''
        INSERT INTO join_add_summary (group_id, total_additions)
        VALUES (?, 1)
        ON CONFLICT(group_id) DO UPDATE SET
        total_additions = total_additions + 1
    ''', (group_id,))

    await db.commit()

async def update_user_leaves(user_id, group_id):
    await db.execute('''
        INSERT INTO user_leaves (user_id, group_id, leave_date)
        VALUES (?, ?, ?)
    ''', (user_id, group_id, datetime.now()))
    await db.commit()






async def update_link_sharing(user_id, group_id):
    await db.execute('''
        INSERT INTO link_sharing (user_id, group_id, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id))
    await db.commit()

async def update_message_deletions(user_id, group_id):
    await db.execute('''
        INSERT INTO message_deletions (user_id, group_id, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id))
    await db.commit()

async def update_message_types(user_id, group_id, message_type):
    await db.execute('''
        INSERT INTO message_types (user_id, group_id, message_type, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, message_type) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, message_type))
    await db.commit()

async def update_engagement(user_id, group_id, period):
    await db.execute('''
        INSERT INTO engagement (user_id, group_id, period, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, period) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, period))
    await db.commit()

async def update_hourly_engagement(user_id, group_id, hour):
    await db.execute('''
        INSERT INTO hourly_engagement (user_id, group_id, hour, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, hour) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, hour))
    await db.commit()

async def update_reaction(user_id, group_id, reaction):
    await db.execute('''
        INSERT INTO reactions (user_id, group_id, reaction, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, reaction) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, reaction))
    await db.commit()

async def update_participation(user_id, group_id, messages, words, characters, replies):
    await db.execute('''
        INSERT INTO participation (user_id, group_id, messages, words, characters, replies)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        messages = participation.messages + excluded.messages,
        words = participation.words + excluded.words,
        characters = participation.characters + excluded.characters,
        replies = participation.replies + excluded.replies
    ''', (user_id, group_id, messages, words, characters, replies))
    await db.commit()

async def update_message_lengths(user_id, group_id, length):
    async with db.execute('SELECT lengths FROM message_lengths WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        current_lengths = await cursor.fetchone()
    if current_lengths:
        lengths = json.loads(current_lengths[0])
        lengths.append(length)
    else:
        lengths = [length]
    await db.execute('''
        INSERT INTO message_lengths (user_id, group_id, lengths)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        lengths = ?
    ''', (user_id, group_id, json.dumps(lengths), json.dumps(lengths)))
    await db.commit()

async def update_media_usage(user_id, group_id, media_type):
    await db.execute('''
        INSERT INTO media_usage (user_id, group_id, media_type, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, media_type) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, media_type))
    await db.commit()

async def update_voice_message_length(user_id, group_id, length):
    await db.execute('''
        INSERT INTO voice_message_lengths (user_id, group_id, length)
        VALUES (?, ?, ?)
    ''', (user_id, group_id, length))
    await db.commit()
    logging.debug(f"Inserted voice message length: user_id={user_id}, group_id={group_id}, length={length}")

async def update_photo_size(user_id, group_id, size, width, height):
    await db.execute('''
        INSERT INTO photo_sizes (user_id, group_id, size, width, height)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, group_id, size, width, height))
    await db.commit()
    logging.debug(f"Inserted photo size: user_id={user_id}, group_id={group_id}, size={size}, width={width}, height={height}")

async def update_video_size(user_id, group_id, size, duration):
    await db.execute('''
        INSERT INTO video_sizes (user_id, group_id, size, duration)
        VALUES (?, ?, ?, ?)
    ''', (user_id, group_id, size, duration))
    await db.commit()
    logging.debug(f"Inserted video size: user_id={user_id}, group_id={group_id}, size={size}, duration={duration}")


async def update_video_selfie_size(user_id, group_id, size, duration):
    await db.execute('''
        INSERT INTO video_selfies (user_id, group_id, size, duration)
        VALUES (?, ?, ?, ?)
    ''', (user_id, group_id, size, duration))
    await db.commit()
    logging.debug(f"Inserted video selfie: user_id={user_id}, group_id={group_id}, size={size}, duration={duration}")


async def update_audio(user_id, group_id, length, size):
    await db.execute('''
        INSERT INTO audios (user_id, group_id, length, size)
        VALUES (?, ?, ?, ?)
    ''', (user_id, group_id, length, size))
    await db.commit()
    logging.debug(f"Inserted audio: user_id={user_id}, group_id={group_id}, length={length}, size={size}")

async def update_forward(user_id, group_id):
    await db.execute('''
        INSERT INTO forwards (user_id, group_id, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id))
    await db.commit()
    logging.debug(f"Updated forward count for user_id={user_id}, group_id={group_id}")

async def save_message_sentiment(user_id, group_id, message_id, sentiment_score):
    logging.debug(f"Saving sentiment: user_id={user_id}, group_id={group_id}, message_id={message_id}, sentiment_score={sentiment_score}")
    
    await db.execute('''
        INSERT OR IGNORE INTO sentiments (user_id, group_id, message_id, sentiment)
        VALUES (?, ?, ?, ?)
    ''', (user_id, group_id, message_id, sentiment_score))

    await db.execute('''
        UPDATE sentiments
        SET sentiment = ?
        WHERE message_id = ?
    ''', (sentiment_score, message_id))

    await db.commit()
    logging.debug(f"Sentiment saved: user_id={user_id}, group_id={group_id}, message_id={message_id}, sentiment_score={sentiment_score}")

async def update_message_edits(user_id, group_id):
    await db.execute('''
        INSERT INTO message_edits (user_id, group_id, edit_count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        edit_count = message_edits.edit_count + 1
    ''', (user_id, group_id))
    await db.commit()

async def update_commands(user_id, group_id, command):
    await db.execute('''
        INSERT INTO commands (user_id, group_id, command, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, command) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, command))
    await db.commit()

async def update_mention(user_id, group_id, mention):
    await db.execute('''
        INSERT INTO mentions (user_id, group_id, mention, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, group_id, mention) DO UPDATE SET
        count = count + 1
    ''', (user_id, group_id, mention))
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
        user = await get_user_entity(user_id)
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
📊 **User Stats for [{user.first_name}](tg://user?id={user_id})({user_id})** 📊

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
                title=f"Stats for {user.first_name}({user_id})",
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
        chat_id = event.chat_id
        
        user = await get_user_entity(user_id)
        results = await asyncio.gather(
            get_user_participation(user_id, chat_id),
            get_user_hourly_engagement(user_id, chat_id),
            get_user_reactions(user_id, chat_id),
            get_user_media_usage(user_id, chat_id),
            get_user_message_lengths(user_id, chat_id),
            get_user_message_edits(user_id, chat_id),
            get_user_commands(user_id, chat_id),
            get_user_mentions(user_id, chat_id),
            get_user_sentiment_analysis(user_id, chat_id),
            get_user_add_stats(user_id, chat_id),
            get_user_join_stats(user_id, chat_id),
            get_user_leave_stats(user_id, chat_id),
            get_user_average_video_size_and_duration(user_id, chat_id),
            get_user_average_photo_size(user_id, chat_id),
            get_user_average_voice_length(user_id, chat_id),
            get_user_average_audio_size_and_duration(user_id, chat_id),
            get_user_average_video_selfie_size_and_duration(user_id, chat_id)
        )

        (participation_stats, hourly_engagement, reactions, media_usage, 
         message_lengths, message_edits, commands, mentions, average_sentiment,
         add_stats, join_stats, leave_stats, 
         (avg_video_size, avg_video_duration), (avg_photo_size, avg_photo_width, avg_photo_height),
         avg_voice_length, (avg_audio_size, avg_audio_duration), (avg_video_selfie_size, avg_video_selfie_duration)) = results

        if average_sentiment is None:
            average_sentiment = 0

        user_stats = f"""
📊 **User Stats for [{user.first_name}](tg://user?id={user_id})({user_id})** 📊

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

👥 **Add Stats:**
{add_stats}

📈 **Join Stats:**
{join_stats}

🚪 **Leave Stats:**
{leave_stats}

🎥 **Average Video Size:** {avg_video_size / (1024*1024):.2f} MB
🎥 **Average Video Duration:** {avg_video_duration:.2f} seconds
📷 **Average Photo Size:** {avg_photo_size / (1024*1024):.2f} MB
📷 **Average Photo Dimensions:** {avg_photo_width:.2f} x {avg_photo_height:.2f}
🎤 **Average Voice Message Length:** {avg_voice_length:.2f} seconds
🎵 **Average Audio Size:** {avg_audio_size / (1024*1024):.2f} MB
🎵 **Average Audio Duration:** {avg_audio_duration:.2f} seconds
🤳 **Average Video Selfie Size:** {avg_video_selfie_size / (1024*1024):.2f} MB
🤳 **Average Video Selfie Duration:** {avg_video_selfie_duration:.2f} seconds
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
    if event.is_group:
        group_id = event.chat_id
        user_id = event.sender_id
        if user_id == bot_user_id:
            return  # Skip the bot's own messages

        message_id = event.message.id
        message_text = event.message.message or ""

        await process_message(event, message_text, user_id, group_id)

def find_urls(text):
    url_pattern = re.compile(
        r'(https?://\S+|www\.\S+)'
    )
    return url_pattern.findall(text)

async def process_message(event, message_text, user_id, group_id):
    await asyncio.gather(
        update_engagement(user_id, group_id, 'daily'),
        update_engagement(user_id, group_id, 'weekly'),
        update_engagement(user_id, group_id, 'monthly'),
        update_hourly_engagement(user_id, group_id, datetime.now().hour),
        process_message_content(event, message_text, user_id, group_id),
        process_media_types(event, user_id, group_id),
        update_message_types(user_id, group_id, 'text' if event.message.media is None else 'media')
    )

    # Detect and update link sharing
    urls = find_urls(message_text)
    if urls:
        await update_link_sharing(user_id, group_id)
        await store_links(user_id, group_id, urls)

    await db.execute('''
        INSERT INTO messages (user_id, group_id, message, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (user_id, group_id, message_text, datetime.now()))
    await db.commit()

    stars = await analyze_sentiment(message_text)
    sentiment_score = (stars / 5) * 100
    await save_message_sentiment(user_id, group_id, event.message.id, sentiment_score)

async def store_links(user_id, group_id, urls):
    for url in urls:
        await db.execute('''
            INSERT INTO links (user_id, group_id, link)
            VALUES (?, ?, ?)
        ''', (user_id, group_id, url))
    await db.commit()

async def process_message_content(event, message_text, user_id, group_id):
    if message_text:
        words = len(message_text.split())
        characters = len(message_text)
        await asyncio.gather(
            update_participation(user_id, group_id, 1, words, characters, 0),
            update_message_lengths(user_id, group_id, characters)
        )

        emojis = emoji_pattern.findall(message_text)
        for emoji in emojis:
            for individual_emoji in list(emoji):
                await update_emoji(individual_emoji)

    if event.message.is_reply:
        await update_participation(user_id, group_id, 0, 0, 0, 1)



# Command handler for /install
@client.on(events.NewMessage(pattern='/install'))
async def install_command_handler(event):
    chat_id = event.chat_id
    user_id = event.sender_id

    # Check if the user issuing the command is an admin
    if not await is_admin(user_id, chat_id):
        await event.reply("You need to be an admin to use this command.")
        return

    total_admins = 0

    async for participant in client.iter_participants(chat_id, filter=ChannelParticipantsAdmins):
        is_owner = isinstance(participant.participant, ChannelParticipantCreator)

        await db.execute('''
            INSERT INTO group_admins (group_id, user_id, is_owner)
            VALUES (?, ?, ?)
            ON CONFLICT(group_id, user_id) DO NOTHING
        ''', (chat_id, participant.id, is_owner))
        total_admins += 1

    await db.commit()
    await event.reply(f"Admins and the owner have been saved into the database. Total admins: {total_admins}.")



@client.on(events.NewMessage)
async def track_forwarded_messages(event):
    if event.message.forward:
        user_id = event.sender_id
        group_id = event.chat_id
        await update_forward(user_id, group_id)

        # Optionally, you can log this event
        logging.debug(f"Forwarded message detected from user_id={user_id}, group_id={group_id}")

async def process_media_types(event, user_id, group_id):
    if event.message.gif:
        await update_media_usage(user_id, group_id, 'gif')
    elif event.message.video:
        if any(isinstance(attr, DocumentAttributeVideo) and attr.round_message for attr in event.message.video.attributes):
            await update_media_usage(user_id, group_id, 'video_selfie')
            file_size = event.message.file.size if event.message.file else 0
            duration = next(attr.duration for attr in event.message.video.attributes if isinstance(attr, DocumentAttributeVideo))
            await update_video_selfie_size(user_id, group_id, file_size, duration)
        else:
            await update_media_usage(user_id, group_id, 'video')
            file_size = event.message.file.size if event.message.file else 0
            duration = event.message.video.attributes[0].duration
            await update_video_size(user_id, group_id, file_size, duration)
    elif event.message.sticker:
        if event.message.sticker.mime_type == "application/x-tgsticker":  # Animated sticker
            await update_media_usage(user_id, group_id, 'animated_sticker')
        else:
            await update_media_usage(user_id, group_id, 'sticker')
    elif event.message.voice:
        await update_media_usage(user_id, group_id, 'voice')
        voice_length = event.message.voice.attributes[0].duration
        await update_voice_message_length(user_id, group_id, voice_length)
    elif event.message.audio:
        await update_media_usage(user_id, group_id, 'audio')
        audio_length = event.message.audio.attributes[0].duration
        file_size = event.message.file.size if event.message.file else 0
        await update_audio(user_id, group_id, audio_length, file_size)
    elif isinstance(event.message.media, MessageMediaPhoto):
        await update_media_usage(user_id, group_id, 'photo')
        file_size = event.message.file.size if event.message.file else 0
        largest_photo = max(event.message.photo.sizes, key=lambda s: s.w * s.h)
        width = largest_photo.w
        height = largest_photo.h
        await update_photo_size(user_id, group_id, file_size, width, height)

@client.on(events.Raw)
async def raw_update_handler(event):
    if isinstance(event, UpdateBotMessageReaction):
        user_id = event.actor.user_id
        if user_id == bot_user_id:
            return  # Skip the bot's own reactions

        new_reactions = event.new_reactions
        chat_id = event.peer.channel_id if hasattr(event.peer, 'channel_id') else event.peer.chat_id

        async def handle_reaction():
            if new_reactions:
                for reaction in new_reactions:
                    reaction_str = reaction.emoticon
                    await update_reaction(user_id, chat_id, reaction_str)

        asyncio.create_task(handle_reaction())

@client.on(events.MessageEdited)
async def track_message_edits(event):
    user_id = event.sender_id
    group_id = event.chat_id

    # Skip the bot's own messages
    if user_id == bot_user_id:
        return

    # Update the edit count in the database
    await db.execute('''
        INSERT INTO message_edits (user_id, group_id, edit_count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, group_id) DO UPDATE SET
        edit_count = edit_count + 1
    ''', (user_id, group_id))
    await db.commit()


@client.on(events.MessageDeleted)
async def track_message_deletions(event):
    print(f"MessageDeleted event received: {event}")
    for msg_id in event.deleted_ids:
        print(f"Processing deletion for message ID: {msg_id}")
        async with db.execute('SELECT user_id, group_id FROM messages WHERE id = ?', (msg_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                user_id = row[0]
                group_id = row[1]  # Ensure you get the correct group_id from the row

                print(f"Found message in DB. User ID: {user_id}, Group ID: {group_id}")

                # Update the deletion count in the database
                await db.execute('''
                    INSERT INTO message_deletions (user_id, group_id, delete_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, group_id) DO UPDATE SET
                    delete_count = delete_count + 1
                ''', (user_id, group_id))
                await db.commit()

                print(f"Updated message deletion count for User ID: {user_id}, Group ID: {group_id}")
            else:
                print(f"Message ID {msg_id} not found in database.")


@client.on(events.ChatAction)
async def handle_chat_action(event):
    print("Received event:", event)
    chat_id = event.chat_id

    if event.action_message:
        action = event.action_message.action
        if isinstance(action, MessageActionChatAddUser):
            added_user_id = action.users[0]
            added_by_user_id = event.action_message.from_id.user_id

            if added_user_id == added_by_user_id:  # User joined by link
                print(f"User {added_user_id} joined by link in chat {chat_id}")
                await update_user_joins(added_user_id, chat_id, 'link')
            else:  # User was added by another user
                print(f"User {added_user_id} added by {added_by_user_id} in chat {chat_id}")
                await update_user_additions(added_user_id, chat_id, added_by_user_id)
        elif isinstance(action, MessageActionChatJoinedByLink):  # User joined via a link
            user_id = event.action_message.from_id.user_id  # Extracting user ID from PeerUser
            print(f"User {user_id} joined by link in chat {chat_id}")
            await update_user_joins(user_id, chat_id, 'link')
        elif isinstance(action, MessageActionChatDeleteUser):  # User left the group
            user_id = action.user_id
            print(f"User {user_id} left the chat {chat_id}")
            await update_user_leaves(user_id, chat_id)
    elif isinstance(event.original_update, UpdateNewChannelMessage):
        update = event.original_update
        action = update.message.action
        user_id = update.message.from_id.user_id
        print(f"UpdateNewChannelMessage action: {action}")

        if isinstance(action, MessageActionChatJoinedByLink):
            print(f"User {user_id} joined by link in chat {chat_id}")
            await update_user_joins(user_id, chat_id, 'link')
        elif isinstance(action, MessageActionChatAddUser):
            added_user_id = action.users[0]
            added_by_user_id = user_id
            if added_user_id == added_by_user_id:  # User joined by link
                print(f"User {added_user_id} joined by link in chat {chat_id}")
                await update_user_joins(added_user_id, chat_id, 'link')
            else:  # User was added by another user
                print(f"User {added_user_id} added by {added_by_user_id} in chat {chat_id}")
                await update_user_additions(added_user_id, chat_id, added_by_user_id)



CATEGORIES = {
    'main': {
        'Engagement Stats': 'engagement_stats',
        'Activity Stats': 'activity_stats',
        'Media Stats': 'media_stats',
        'Sentiment Stats': 'sentiment_stats',
        'User Join/Add/Leave Stats': 'user_stats',
        'My Stats': 'my_stats'
    }
}

def create_buttons(category):
    buttons = []
    items = list(CATEGORIES[category].items())
    for label, data in items:
        buttons.append([Button.inline(label, data.encode('utf-8'))])
    return buttons

def split_into_pages(content, page_size=4096):
    pages = []
    while len(content) > page_size:
        split_index = content.rfind('\n', 0, page_size)
        if split_index == -1:
            split_index = page_size
        pages.append(content[:split_index])
        content = content[split_index:]
    pages.append(content)
    return pages

@client.on(events.NewMessage(pattern='/groupstats'))
async def group_stats_command_handler(event):
    chat_id = event.chat_id
    user_id = event.sender_id

    if not await is_admin_in_db(chat_id, user_id):
        await event.reply("You must be an admin to use this command.")
        return

    join_stats = await get_join_stats(chat_id)
    addition_stats = await get_addition_stats(chat_id)
    leave_stats = await get_leave_stats(chat_id)
    join_add_summary = await get_join_add_summary(chat_id)

    result = f"""
📊 **Group Stats** 📊

{join_add_summary}

{join_stats}

{addition_stats}

{leave_stats}
"""
    buttons = create_buttons('main')
    await event.reply(result, buttons=buttons)

@client.on(events.CallbackQuery)
async def callback_query_handler(event):
    chat_id = event.chat_id
    user_id = event.sender_id
    data = event.data.decode('utf-8').split('|')

    if data[0] == 'back_to_main':
        buttons = create_buttons('main')
        await event.edit("📊 **Group Stats** 📊\nChoose a category to view:", buttons=buttons)
        return

    if data[0] in CATEGORIES:
        buttons = create_buttons(data[0])
        await event.edit(f"📊 **{data[0].replace('_stats', '').capitalize()} Stats** 📊\nChoose a stat to view:", buttons=buttons)
        return

    if data[0] == 'next_page' or data[0] == 'prev_page':
        category = data[1]
        page_number = int(data[2])
        result = await fetch_category_stats(category, chat_id, page_number)
        buttons = [[Button.inline("Previous Page", f"prev_page|{category}|{page_number-1}".encode('utf-8'))]] if page_number > 1 else []
        buttons.append([Button.inline("Next Page", f"next_page|{category}|{page_number+1}".encode('utf-8'))])
        buttons.append([Button.inline("Back", "back_to_main".encode('utf-8'))])
        await event.edit(result, buttons=buttons)
        return

    if data[0] == 'my_stats':
        result = await fetch_user_stats(user_id, chat_id)
        buttons = [[Button.inline("Back", "back_to_main".encode('utf-8'))]]
        await event.edit(result, buttons=buttons)
        return

    result = await fetch_category_stats(data[0], chat_id)
    buttons = [[Button.inline("Back", "back_to_main".encode('utf-8'))]]
    current_message = await event.get_message()
    current_text = current_message.text

    if current_text != result:
        await event.edit(result, buttons=buttons)
    else:
        print("Content of the message was not modified, skipping edit to avoid MessageNotModifiedError.")

async def fetch_user_stats(user_id, chat_id):
    try:
        participation_stats = await get_user_participation(user_id, chat_id)
        hourly_engagement = await get_user_hourly_engagement(user_id, chat_id)
        reactions = await get_user_reactions(user_id, chat_id)
        media_usage = await get_user_media_usage(user_id, chat_id)
        message_lengths = await get_user_message_lengths(user_id, chat_id)
        message_edits = await get_user_message_edits(user_id, chat_id)
        commands = await get_user_commands(user_id, chat_id)
        mentions = await get_user_mentions(user_id, chat_id)
        average_sentiment = await get_user_sentiment_analysis(user_id, chat_id)
        avg_video_size, avg_video_duration = await get_user_average_video_size_and_duration(user_id, chat_id)
        avg_photo_size, avg_photo_width, avg_photo_height = await get_user_average_photo_size(user_id, chat_id)
        avg_voice_length = await get_user_average_voice_length(user_id, chat_id)
        avg_audio_size, avg_audio_duration = await get_user_average_audio_size_and_duration(user_id, chat_id)
        avg_video_selfie_size, avg_video_selfie_duration = await get_user_average_video_selfie_size_and_duration(user_id, chat_id)

        user_stats = f"""
📊 **User Stats for [{user_id}](tg://user?id={user_id})** 📊

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

🎥 **Average Video Size:** {avg_video_size / (1024*1024):.2f} MB
🎥 **Average Video Duration:** {avg_video_duration:.2f} seconds
📷 **Average Photo Size:** {avg_photo_size / (1024*1024):.2f} MB
📷 **Average Photo Dimensions:** {avg_photo_width:.2f} x {avg_photo_height:.2f}
🎤 **Average Voice Message Length:** {avg_voice_length:.2f} seconds
🎵 **Average Audio Size:** {avg_audio_size / (1024*1024):.2f} MB
🎵 **Average Audio Duration:** {avg_audio_duration:.2f} seconds
🤳 **Average Video Selfie Size:** {avg_video_selfie_size / (1024*1024):.2f} MB
🤳 **Average Video Selfie Duration:** {avg_video_selfie_duration:.2f} seconds
"""
        return user_stats
    except Exception as e:
        return f"An error occurred while fetching the data: {str(e)}"

# Other existing functions...

async def fetch_category_stats(category, chat_id, page_number=1):
    try:
        if category == 'user_stats':
            join_stats = await get_join_stats(chat_id)
            addition_stats = await get_addition_stats(chat_id)
            leave_stats = await get_leave_stats(chat_id)
            join_add_summary = await get_join_add_summary(chat_id)
            content = (
                f"📊 **User Join/Add/Leave Stats** 📊\n\n"
                f"{join_add_summary}\n"
                f"{join_stats}\n"
                f"{addition_stats}\n"
                f"{leave_stats}\n"
            )
        elif category == 'engagement_stats':
            results = await asyncio.gather(
                format_engagement_leaderboard(chat_id, 'daily'),
                format_engagement_leaderboard(chat_id, 'weekly'),
                format_engagement_leaderboard(chat_id, 'monthly'),
                format_reaction_leaderboard(chat_id),
                format_popular_reactions(chat_id)
            )
            daily_leaderboard, weekly_leaderboard, monthly_leaderboard, reaction_leaderboard, popular_reactions = results
            content = (
                f"📊 **Engagement Stats** 📊\n\n"
                f"{daily_leaderboard}\n"
                f"{weekly_leaderboard}\n"
                f"{monthly_leaderboard}\n"
                f"{reaction_leaderboard}\n"
                f"{popular_reactions}\n"
            )
        elif category == 'activity_stats':
            results = await asyncio.gather(
                format_active_hours(chat_id),
                format_participation_stats(chat_id),
                format_message_length_stats(chat_id)
            )
            active_hours, participation_stats, message_length_stats = results
            content = (
                f"📊 **Activity Stats** 📊\n\n"
                f"{active_hours}\n"
                f"{participation_stats}\n"
                f"{message_length_stats}\n"
            )
        elif category == 'media_stats':
            results = await asyncio.gather(
                get_media_type_distribution(chat_id),
                format_emoji_stats(chat_id),
                format_forward_stats(chat_id),
                get_link_sharing_stats(chat_id),
                get_message_deletion_stats(chat_id),
                calculate_average_video_size_and_duration(chat_id),
                calculate_average_photo_size(chat_id),
                calculate_average_voice_length(chat_id),
                calculate_average_audio_size_and_duration(chat_id),
                calculate_average_video_selfie_size_and_duration(chat_id),
                get_top_media_users(chat_id, 'gif'),
                get_top_media_users(chat_id, 'video'),
                get_top_media_users(chat_id, 'photo'),
                get_top_media_users(chat_id, 'voice'),
                get_top_media_users(chat_id, 'audio'),
                get_top_media_users(chat_id, 'animated_sticker'),
                get_top_media_users(chat_id, 'sticker')
            )
            (media_type_distribution, emoji_stats, forward_stats, link_sharing_stats, message_deletion_stats,
             (avg_video_size, avg_video_duration), (avg_photo_size, avg_photo_width, avg_photo_height),
             avg_voice_length, (avg_audio_size, avg_audio_duration), (avg_video_selfie_size, avg_video_selfie_duration),
             top_gif_users, top_video_users, top_photo_users, top_voice_users, top_audio_users, top_animated_sticker_users, top_sticker_users) = results

            top_gif = format_top_media_users('GIF', top_gif_users)
            top_video = format_top_media_users('Video', top_video_users)
            top_photo = format_top_media_users('Photo', top_photo_users)
            top_voice = format_top_media_users('Voice', top_voice_users)
            top_audio = format_top_media_users('Audio', top_audio_users)
            top_animated_sticker = format_top_media_users('Animated Sticker', top_animated_sticker_users)
            top_sticker = format_top_media_users('Sticker', top_sticker_users)

            content = (
                f"📊 **Media Stats** 📊\n\n"
                f"{media_type_distribution}\n"
                f"{emoji_stats}\n"
                f"{forward_stats}\n"
                f"{link_sharing_stats}\n"
                f"{message_deletion_stats}\n"
                f"🎥 **Average Video Size:** {avg_video_size / (1024*1024):.2f} MB\n"
                f"🎥 **Average Video Duration:** {avg_video_duration:.2f} seconds\n"
                f"📷 **Average Photo Size:** {avg_photo_size / (1024*1024):.2f} MB\n"
                f"📷 **Average Photo Dimensions:** {avg_photo_width:.2f} x {avg_photo_height:.2f}\n"
                f"🎤 **Average Voice Message Length:** {avg_voice_length:.2f} seconds\n"
                f"🎵 **Average Audio Size:** {avg_audio_size / (1024*1024):.2f} MB\n"
                f"🎵 **Average Audio Duration:** {avg_audio_duration:.2f} seconds\n"
                f"🤳 **Average Video Selfie Size:** {avg_video_selfie_size / (1024*1024):.2f} MB\n"
                f"🤳 **Average Video Selfie Duration:** {avg_video_selfie_duration:.2f} seconds\n\n"
                f"{top_gif}\n\n"
                f"{top_video}\n\n"
                f"{top_photo}\n\n"
                f"{top_voice}\n\n"
                f"{top_audio}\n\n"
                f"{top_animated_sticker}\n\n"
                f"{top_sticker}\n\n"
            )
        elif category == 'sentiment_stats':
            results = await asyncio.gather(
                calculate_group_sentiment(chat_id),
                calculate_average_edits_per_message(chat_id),
                get_top_sentiment_users(chat_id)
            )
            sentiment_data, edits_per_message, top_sentiment_users = results
            group_sentiment, most_negative_user, most_positive_user = sentiment_data
            top_positive_users, top_negative_users = top_sentiment_users
            
            top_positive = "\n".join([f"- [{user_id}](tg://user?id={user_id}) ({sentiment:.2f}%)" for user_id, sentiment in top_positive_users])
            top_negative = "\n".join([f"- [{user_id}](tg://user?id={user_id}) ({sentiment:.2f}%)" for user_id, sentiment in top_negative_users])

            content = (
                f"📊 **Sentiment Stats** 📊\n\n"
                f"😊 **Average Group Sentiment:** {group_sentiment:.2f}%\n"
                f"🔴 **Top 3 Negative Users:**\n{top_negative}\n"
                f"🟢 **Top 3 Positive Users:**\n{top_positive}\n"
                f"✏️ **Edits Per Message:** {edits_per_message:.2f}\n"
            )
        else:
            return "Invalid selection."

        pages = split_into_pages(content)
        page_index = page_number - 1

        if page_index < 0 or page_index >= len(pages):
            return "Invalid page number."

        return pages[page_index]

    except Exception as e:
        return f"An error occurred while fetching the data: {str(e)}"









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
        format_media_stats('animated_sticker'),
        format_media_stats('voice'),
        format_media_stats('audio'),
        format_emoji_stats(),
        format_forward_stats()
    )

    (daily_leaderboard, weekly_leaderboard, monthly_leaderboard, active_hours, 
     reaction_leaderboard, popular_reactions, participation_stats, 
     message_length_stats, gif_stats, video_stats, sticker_stats, 
     animated_sticker_stats, voice_stats, audio_stats, emoji_stats, 
     forward_stats) = results

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
        f"{animated_sticker_stats}\n"
        f"{voice_stats}\n"
        f"{audio_stats}\n"
        f"{emoji_stats}\n"
        f"{forward_stats}\n"
    )

    return message



# Define the function to get user add stats
# Define the function to get user add stats
async def get_user_add_stats(user_id, chat_id):
    async with db.execute('''
        SELECT COUNT(*) FROM user_additions WHERE added_by_user_id = ? AND group_id = ?
    ''', (user_id, chat_id)) as cursor:
        row = await cursor.fetchone()
    return f"Added Users: {row[0]}" if row else "No add stats data."

# Define the function to get user join stats
async def get_user_join_stats(user_id, chat_id):
    async with db.execute('''
        SELECT COUNT(*) FROM user_joins WHERE user_id = ? AND group_id = ?
    ''', (user_id, chat_id)) as cursor:
        row = await cursor.fetchone()
    return f"Joins: {row[0]}" if row else "No join stats data."

# Define the function to get user leave stats
async def get_user_leave_stats(user_id, chat_id):
    async with db.execute('''
        SELECT COUNT(*) FROM user_leaves WHERE user_id = ? AND group_id = ?
    ''', (user_id, chat_id)) as cursor:
        row = await cursor.fetchone()
    return f"Leaves: {row[0]}" if row else "No leave stats data."

# Define the function to get user average video size and duration
async def get_user_average_video_size_and_duration(user_id, group_id):
    async with db.execute('SELECT AVG(size), AVG(duration) FROM video_sizes WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0

# Define the function to get user average photo size
async def get_user_average_photo_size(user_id, group_id):
    async with db.execute('SELECT AVG(size), AVG(width), AVG(height) FROM photo_sizes WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0, row[2] if row[2] else 0

# Define the function to get user average voice length
async def get_user_average_voice_length(user_id, group_id):
    async with db.execute('SELECT AVG(length) FROM voice_message_lengths WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0

# Define the function to get user average audio size and duration
async def get_user_average_audio_size_and_duration(user_id, group_id):
    async with db.execute('SELECT AVG(size), AVG(length) FROM audios WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0

# Define the function to get user average video selfie size and duration
async def get_user_average_video_selfie_size_and_duration(user_id, group_id):
    async with db.execute('SELECT AVG(size), AVG(duration) FROM video_selfies WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0




async def get_top_media_users(group_id, media_type):
    async with db.execute('''
        SELECT user_id, count 
        FROM media_usage 
        WHERE group_id = ? AND media_type = ? 
        ORDER BY count DESC 
        LIMIT 3
    ''', (group_id, media_type)) as cursor:
        rows = await cursor.fetchall()
    return rows

async def get_join_stats(group_id):
    async with db.execute('''
        SELECT join_method, COUNT(*) FROM user_joins WHERE group_id = ? GROUP BY join_method
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "📈 **Join Stats** 📈\n"
    for join_method, count in rows:
        result += f"{join_method.capitalize()}: {count} users\n"
    return result if rows else "No join data available."

async def get_addition_stats(group_id):
    async with db.execute('''
        SELECT added_by_user_id, COUNT(*) FROM user_additions WHERE group_id = ? GROUP BY added_by_user_id ORDER BY COUNT(*) DESC LIMIT 10
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "👥 **Top 10 Users Adding Members** 👥\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} users\n"
    return result if rows else "No addition data available."

async def get_leave_stats(group_id):
    async with db.execute('''
        SELECT COUNT(*) FROM user_leaves WHERE group_id = ?
    ''', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return f"🚪 **Users Left** 🚪\nTotal: {row[0]} users" if row else "No leave data available."

async def get_join_add_summary(group_id):
    async with db.execute('''
        SELECT total_joins, joins_by_link, total_additions FROM join_add_summary WHERE group_id = ?
    ''', (group_id,)) as cursor:
        row = await cursor.fetchone()
    
    if row:
        total_joins = row[0] if row[0] else 0
        joins_by_link = row[1] if row[1] else 0
        total_additions = row[2] if row[2] else 0
    else:
        total_joins = 0
        joins_by_link = 0
        total_additions = 0

    return (f"📊 **Join/Add Summary** 📊\n"
            f"Total Joins: {total_joins} users\n"
            f"Joins by Link: {joins_by_link} users\n"
            f"Total Additions: {total_additions} users")


async def get_user_sentiment_scores():
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

    sentiment_scores.sort(key=lambda x: x[1])

    most_negative_user = sentiment_scores[0]
    most_positive_user = sentiment_scores[-1]

    return most_negative_user, most_positive_user

async def generate_sentiment_report():
    most_negative_user, most_positive_user = await get_extreme_sentiment_users()

    if most_negative_user and most_positive_user:
        most_negative_name = await get_user_entity(most_negative_user[0])
        most_positive_name = await get_user_entity(most_positive_user[0])
        report = f"""
📊 **User Sentiment Report** 📊

🔴 **Most Negative User**:
- [{most_negative_name.first_name}](tg://user?id={most_negative_user[0]})({most_negative_user[0]})
- Average Sentiment Score: {most_negative_user[1]:.2f}

🟢 **Most Positive User**:
- [{most_positive_name.first_name}](tg://user?id={most_positive_user[0]})({most_positive_user[0]})
- Average Sentiment Score: {most_positive_user[1]:.2f}
"""
    else:
        report = "No sentiment data available."

    return report






async def calculate_average_video_selfie_size_and_duration(group_id):
    async with db.execute('SELECT AVG(size), AVG(duration) FROM video_selfies WHERE group_id = ?', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0

async def get_top_users_by_video_selfies(group_id):
    async with db.execute('''
        SELECT user_id, COUNT(*) as count
        FROM video_selfies
        WHERE group_id = ?
        GROUP BY user_id
        ORDER BY count DESC
        LIMIT 10
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    
    result = "📷 **Top 10 Users by Video Selfies** 📷\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} video selfies\n"
    return result if rows else "No video selfies data available."



async def calculate_average_audio_size_and_duration(group_id):
    async with db.execute('SELECT AVG(size), AVG(length) FROM audios WHERE group_id = ?', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0



async def format_top_sentiment_users(group_id, sentiment_type):
    order = 'ASC' if sentiment_type == 'negative' else 'DESC'
    async with db.execute(f'''
        SELECT user_id, AVG(sentiment) as avg_sentiment
        FROM sentiments
        WHERE group_id = ?
        GROUP BY user_id
        ORDER BY avg_sentiment {order}
        LIMIT 10
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()

    result = ""
    for rank, (user_id, avg_sentiment) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {avg_sentiment:.2f}%\n"

    return result if rows else f"No {sentiment_type} sentiment data available."



async def calculate_average_edits_per_message(group_id):
    async with db.execute('SELECT AVG(edit_count) FROM message_edits WHERE group_id = ?', (group_id,)) as cursor:
        average_edits = await cursor.fetchone()
    return average_edits[0] if average_edits and average_edits[0] is not None else 0



async def calculate_group_sentiment(group_id):
    async with db.execute('SELECT user_id, AVG(sentiment) as avg_sentiment FROM sentiments WHERE group_id = ? GROUP BY user_id', (group_id,)) as cursor:
        rows = await cursor.fetchall()

    if rows:
        sentiments = [row[1] for row in rows]
        average_sentiment = sum(sentiments) / len(sentiments)

        rows.sort(key=lambda x: x[1])
        most_negative_user = rows[0]
        most_positive_user = rows[-1]

        return average_sentiment, most_negative_user, most_positive_user
    return 0.0, (None, 0.0), (None, 0.0)



async def get_top_sentiment_users(group_id):
    async with db.execute('''
        SELECT user_id, AVG(sentiment) as avg_sentiment 
        FROM sentiments 
        WHERE group_id = ? 
        GROUP BY user_id 
        ORDER BY avg_sentiment DESC 
        LIMIT 3
    ''', (group_id,)) as cursor:
        positive_users = await cursor.fetchall()

    async with db.execute('''
        SELECT user_id, AVG(sentiment) as avg_sentiment 
        FROM sentiments 
        WHERE group_id = ? 
        GROUP BY user_id 
        ORDER BY avg_sentiment ASC 
        LIMIT 3
    ''', (group_id,)) as cursor:
        negative_users = await cursor.fetchall()

    return positive_users, negative_users



async def get_media_type_distribution(group_id):
    async with db.execute('''
        SELECT media_type, SUM(count) as total_count 
        FROM media_usage 
        WHERE user_id IN (SELECT DISTINCT user_id FROM messages WHERE group_id = ?)
        GROUP BY media_type
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()

    result = "🎥 **Media Type Distribution** 🎥\n"
    for media_type, total_count in rows:
        result += f"{media_type.capitalize()}: {total_count} uses\n"


    return result if rows else "No media usage data available."

async def get_total_link_count():
    async with db.execute('SELECT COUNT(*) FROM links') as cursor:
        row = await cursor.fetchone()
    return row[0] if row else 0

async def get_general_sentiment():
    async with db.execute('SELECT sentiment FROM sentiments') as cursor:
        rows = await cursor.fetchall()
    
    if rows:
        sentiments = [row[0] for row in rows]
        general_sentiment = sum(sentiments) / len(sentiments)
        return general_sentiment
    return None

async def analyze_sentiment(message_text):
    result = sentiment_analyzer(message_text)
    label = result[0]['label']
    stars = int(label[0])
    return stars


async def get_link_sharing_stats(group_id):
    async with db.execute('SELECT user_id, count FROM link_sharing WHERE group_id = ? ORDER BY count DESC LIMIT 10', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "🔗 **Top 10 Users by Link Sharing** 🔗\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} links\n"
    return result if rows else "No link sharing data available."

async def get_message_deletion_stats(group_id):
    async with db.execute('SELECT user_id, count FROM message_deletions WHERE group_id = ? ORDER BY count DESC LIMIT 10', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "❌ **Top 10 Users by Message Deletions** ❌\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} deletions\n"
    return result if rows else "No message deletion data available."

async def calculate_average_edits():
    async with db.execute('SELECT AVG(edit_count) FROM message_edits') as cursor:
        average_edits = await cursor.fetchone()
    logging.debug(f"Average message edits calculation result: {average_edits}")
    return average_edits[0] if average_edits and average_edits[0] is not None else 0

async def get_most_active_users():
    async with db.execute('SELECT user_id, messages FROM participation ORDER BY messages DESC LIMIT 10') as cursor:
        rows = await cursor.fetchall()
    result = "📈 **Top 10 Most Active Users** 📈\n"
    for rank, (user_id, message_count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {message_count} messages\n"
    return result

async def get_engagement_by_time_of_day():
    async with db.execute('SELECT hour, SUM(count) as message_count FROM hourly_engagement GROUP BY hour ORDER BY message_count DESC') as cursor:
        rows = await cursor.fetchall()
    result = "⏰ **Engagement by Time of Day** ⏰\n"
    for hour, message_count in rows:
        result += f"{hour}:00 - {message_count} messages\n"
    return result


async def format_engagement_leaderboard(group_id, period):
    async with db.execute('SELECT user_id, count FROM engagement WHERE group_id = ? AND period = ? ORDER BY count DESC LIMIT 10', (group_id, period)) as cursor:
        rows = await cursor.fetchall()
    result = f"🏆 **Top 10 Active Users ({period.capitalize()})** 🏆\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} messages\n"
    return result

async def format_active_hours(group_id):
    async with db.execute('SELECT hour, SUM(count) FROM hourly_engagement WHERE group_id = ? GROUP BY hour ORDER BY SUM(count) DESC LIMIT 10', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "⏰ **Top 10 Active Hours** ⏰\n"
    for rank, (hour, count) in enumerate(rows, start=1):
        result += f"{rank}. {hour}:00 - {count} messages\n"
    return result


def format_top_media_users(title, users):
    return (f"📈 **Top 3 Users by {title}** 📈\n" +
            "\n".join([f"- [{user_id}](tg://user?id={user_id}) ({count} {title.lower()})" for user_id, count in users])) if users else f"No {title.lower()} data available."




async def format_reaction_leaderboard(group_id):
    async with db.execute('''
        SELECT user_id, SUM(count) as total_reactions 
        FROM reactions 
        WHERE user_id IN (SELECT DISTINCT user_id FROM messages WHERE group_id = ?)
        GROUP BY user_id 
        ORDER BY total_reactions DESC 
        LIMIT 10
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    
    result = "🏆 **Top 10 Users by Reactions Received** 🏆\n"
    for rank, (user_id, total_reactions) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {total_reactions} reactions\n"
    
    return result if rows else "No reactions data available."

async def format_popular_reactions(group_id):
    async with db.execute('''
        SELECT reaction, SUM(count) as total_count 
        FROM reactions 
        WHERE user_id IN (SELECT DISTINCT user_id FROM messages WHERE group_id = ?)
        GROUP BY reaction 
        ORDER BY total_count DESC 
        LIMIT 10
    ''', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    
    result = "🎉 **Top 10 Popular Reactions** 🎉\n"
    for rank, (reaction, total_count) in enumerate(rows, start=1):
        result += f"{rank}. {reaction} - {total_count} times\n"
    
    return result if rows else "No popular reactions data available."


async def format_participation_stats(group_id):
    async with db.execute('SELECT user_id, messages, words, characters, replies FROM participation WHERE group_id = ? ORDER BY messages DESC LIMIT 10', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "📈 **Top 10 Participation Stats** 📈\n"
    for rank, (user_id, messages, words, characters, replies) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += (f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - "
                   f"{messages} messages, "
                   f"{words} words, "
                   f"{characters} characters, "
                   f"{replies} replies\n")
    return result

async def format_message_length_stats(group_id):
    async with db.execute('SELECT user_id, lengths FROM message_lengths WHERE group_id = ?', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    sorted_data = sorted(
        ((user_id, sum(json.loads(lengths)) / len(json.loads(lengths)) if json.loads(lengths) else 0) for user_id, lengths in rows),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    result = "📝 **Top 10 Average Message Lengths** 📝\n"
    for rank, (user_id, avg_length) in enumerate(sorted_data, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {avg_length:.2f} characters/message\n"
    return result

async def format_media_stats(group_id, media_type):
    async with db.execute('SELECT user_id, count FROM media_usage WHERE group_id = ? AND media_type = ? ORDER BY count DESC LIMIT 10', (group_id, media_type)) as cursor:
        rows = await cursor.fetchall()
    result = f"**{media_type.capitalize()} Stats (Top 10 Users):**\n"
    for user_id, count in rows:
        user = await get_user_entity(user_id)
        result += f" - [{user.first_name}](tg://user?id={user_id})({user_id}): {count} {media_type}s\n"
    return result

async def format_emoji_stats(group_id):
    async with db.execute('SELECT emoji, count FROM emojis ORDER BY count DESC LIMIT 10') as cursor:
        rows = await cursor.fetchall()
    result = "**Emoji Stats (Top 10 Emojis):**\n"
    for emoji, count in rows:
        result += f" - {emoji}: {count} times\n"
    return result

async def format_forward_stats(group_id):
    async with db.execute('SELECT user_id, count FROM forwards WHERE group_id = ? ORDER BY count DESC LIMIT 10', (group_id,)) as cursor:
        rows = await cursor.fetchall()
    result = "📤 **Top 10 Users by Forwards** 📤\n"
    for rank, (user_id, count) in enumerate(rows, start=1):
        user = await get_user_entity(user_id)
        result += f"{rank}. [{user.first_name}](tg://user?id={user_id})({user_id}) - {count} forwards\n"
    return result

async def calculate_average_voice_length(group_id):
    async with db.execute('SELECT AVG(length) FROM voice_message_lengths WHERE group_id = ?', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0

async def calculate_average_photo_size(group_id):
    async with db.execute('SELECT AVG(size), AVG(width), AVG(height) FROM photo_sizes WHERE group_id = ?', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0, row[2] if row[2] else 0


async def calculate_average_video_size_and_duration(group_id):
    async with db.execute('SELECT AVG(size), AVG(duration) FROM video_sizes WHERE group_id = ?', (group_id,)) as cursor:
        row = await cursor.fetchone()
    return row[0] if row[0] else 0, row[1] if row[1] else 0


async def get_user_participation(user_id, group_id):
    async with db.execute('SELECT messages, words, characters, replies FROM participation WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    if row:
        return f"Messages: {row[0]}\nWords: {row[1]}\nCharacters: {row[2]}\nReplies: {row[3]}"
    return "No participation data."

async def get_user_hourly_engagement(user_id, group_id):
    async with db.execute('SELECT hour, count FROM hourly_engagement WHERE user_id = ? AND group_id = ? ORDER BY hour', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    result = ""
    for hour, count in rows:
        result += f"{hour}:00 - {count} messages\n"
    return result if result else "No hourly engagement data."

async def get_user_reactions(user_id, group_id):
    async with db.execute('SELECT reaction, count FROM reactions WHERE user_id = ? AND group_id = ? ORDER BY count DESC', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    result = ""
    for reaction, count in rows:
        result += f"{reaction} - {count} times\n"
    return result if result else "No reaction data."

async def get_user_media_usage(user_id, group_id):
    async with db.execute('SELECT media_type, count FROM media_usage WHERE user_id = ? AND group_id = ? ORDER BY count DESC', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    result = ""
    for media_type, count in rows:
        result += f"{media_type.capitalize()}: {count} uses\n"
    return result if result else "No media usage data."

async def get_user_sentiment_analysis(user_id, group_id):
    async with db.execute('SELECT sentiment FROM sentiments WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    
    if rows:
        sentiments = [row[0] for row in rows]
        average_sentiment = sum(sentiments) / len(sentiments)
        return average_sentiment
    return None

async def get_user_message_lengths(user_id, group_id):
    async with db.execute('SELECT lengths FROM message_lengths WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    if row:
        lengths = json.loads(row[0])
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        return f"Average Length: {avg_length:.2f} characters"
    return "No message length data."

async def get_user_message_edits(user_id, group_id):
    async with db.execute('SELECT edit_count FROM message_edits WHERE user_id = ? AND group_id = ?', (user_id, group_id)) as cursor:
        row = await cursor.fetchone()
    return f"Message Edits: {row[0]}" if row else "No message edits data."

async def get_user_commands(user_id, group_id):
    async with db.execute('SELECT command, count FROM commands WHERE user_id = ? AND group_id = ? ORDER BY count DESC', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    result = ""
    for command, count in rows:
        result += f"{command}: {count} times\n"
    return result if result else "No commands data."

async def get_user_mentions(user_id, group_id):
    async with db.execute('SELECT mention, count FROM mentions WHERE user_id = ? AND group_id = ? ORDER BY count DESC', (user_id, group_id)) as cursor:
        rows = await cursor.fetchall()
    result = ""
    for mention, count in rows:
        result += f"{mention}: {count} times\n"
    return result if result else "No mentions data."

async def fetch_daily_message_counts():
    async with db.execute('SELECT date(timestamp) as date, COUNT(*) as message_count FROM messages GROUP BY date ORDER BY date') as cursor:
        rows = await cursor.fetchall()
    return rows



@client.on(events.NewMessage(pattern='/dump'))
async def dump_replied_message(event):
    if event.is_reply:
        reply_message = await event.get_reply_message()
        print(reply_message)
        message_id = reply_message.id
        sender_id = reply_message.sender_id
        timestamp = reply_message.date
        message_content = reply_message.message or "No text content"
        media = reply_message.media
        media_info = f"Media: {media}" if media else "Media: None"
        reply_to_msg_id = reply_message.reply_to_msg_id
        reply_info = f"Replied to Message ID: {reply_to_msg_id}" if reply_to_msg_id else "Replied to Message ID: None"
        views = reply_message.views
        views_info = f"Views: {views}" if views else "Views: Not available"
        forwards = reply_message.forwards
        forwards_info = f"Forwards: {forwards}" if forwards else "Forwards: Not available"
        edit_date = reply_message.edit_date
        edit_info = f"Edited on: {edit_date}" if edit_date else "Edited on: Not edited"

        dump_info = (
            f"Replied Message Dump\n"
            f"Message ID: {message_id}\n"
            f"Sender ID: {sender_id}\n"
            f"Timestamp: {timestamp}\n"
            f"Content: {message_content}\n"
            f"{media_info}\n"
            f"{reply_info}\n"
            f"{views_info}\n"
            f"{forwards_info}\n"
            f"{edit_info}\n"
        )
        
        await event.reply(reply_message)
    else:
        await event.reply("Please reply to a message to dump its content.")






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
    return model, forecast

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
