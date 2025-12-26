import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import redis
from openai import OpenAI
from dotenv import load_dotenv
import jwt
from typing import Optional, List
import uvicorn

load_dotenv()

app = FastAPI(
    title="AI Customer Support Chatbot",
    description="Full-featured customer support system with AI assistant",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Redis Connection
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True
    )
    redis_client.ping()
    print("Redis connected successfully")
except Exception as e:
    print(f" Redis connection failed: {e}")
    redis_client = None

# GitHub AI Model
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN:
    ai_client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=GITHUB_TOKEN
    )
    print("GitHub AI client initialized")
else:
    print(" GITHUB_TOKEN not found, using mock responses")
    ai_client = None

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# ========== MODELS ==========
class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class Ticket(BaseModel):
    id: Optional[str] = None
    subject: str
    description: str
    priority: str = "medium"
    category: str = "general"
    status: str = "open"
    user_email: str
    created_by: str

class TicketUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None

class FAQItem(BaseModel):
    id: Optional[str] = None
    question: str
    answer: str
    category: str = "general"

class KnowledgeBaseItem(BaseModel):
    id: Optional[str] = None
    title: str
    content: str
    category: str = "general"
    tags: List[str] = []

class ChatHistoryRequest(BaseModel):
    user_id: str
    limit: Optional[int] = 50

# ========== DATABASE FUNCTIONS ==========
def create_user(user_data: dict):
    """Create new user in Redis"""
    if not redis_client:
        return None
    
    user_id = f"user:{user_data['email']}"
    if redis_client.exists(user_id):
        return None
    
    user_data.update({
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    })
    redis_client.hset(user_id, mapping=user_data)
    return user_data

def authenticate_user(email: str, password: str):
    """Authenticate user"""
    if not redis_client:
        return None
    
    user_id = f"user:{email}"
    user_data = redis_client.hgetall(user_id)
    if user_data and user_data.get("password") == password:
        # Update last login
        redis_client.hset(user_id, "last_login", datetime.now().isoformat())
        return user_data
    return None

def create_jwt_token(user_email: str):
    """Create JWT token"""
    payload = {
        "sub": user_email,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except:
        return None

def save_to_database(data_type: str, key: str, data: dict):
    """Save any data to Redis database"""
    if not redis_client:
        return False
    
    try:
        # Create composite key
        full_key = f"{data_type}:{key}"
        
        # Add timestamp
        data['_created_at'] = datetime.now().isoformat()
        data['_updated_at'] = datetime.now().isoformat()
        
        # Save as hash
        redis_client.hset(full_key, mapping=data)
        
        # Add to index
        index_key = f"index:{data_type}"
        redis_client.sadd(index_key, key)
        
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def get_from_database(data_type: str, key: str):
    """Get data from Redis database"""
    if not redis_client:
        return None
    
    try:
        full_key = f"{data_type}:{key}"
        data = redis_client.hgetall(full_key)
        return data if data else None
    except Exception as e:
        print(f"Error getting from database: {e}")
        return None

def get_all_from_database(data_type: str):
    """Get all items of a specific type from database"""
    if not redis_client:
        return []
    
    try:
        index_key = f"index:{data_type}"
        keys = redis_client.smembers(index_key)
        
        items = []
        for key in keys:
            full_key = f"{data_type}:{key}"
            data = redis_client.hgetall(full_key)
            if data:
                items.append({
                    'id': key,
                    'data': data
                })
        
        return items
    except Exception as e:
        print(f"Error getting all from database: {e}")
        return []

def update_in_database(data_type: str, key: str, data: dict):
    """Update data in Redis database"""
    if not redis_client:
        return False
    
    try:
        full_key = f"{data_type}:{key}"
        
        if not redis_client.exists(full_key):
            return False
        
        # Update only provided fields
        for field, value in data.items():
            redis_client.hset(full_key, field, value)
        
        # Update timestamp
        redis_client.hset(full_key, '_updated_at', datetime.now().isoformat())
        
        return True
    except Exception as e:
        print(f"Error updating in database: {e}")
        return False

def delete_from_database(data_type: str, key: str):
    """Delete data from Redis database"""
    if not redis_client:
        return False
    
    try:
        full_key = f"{data_type}:{key}"
        
        if not redis_client.exists(full_key):
            return False
        
        # Remove from index
        index_key = f"index:{data_type}"
        redis_client.srem(index_key, key)
        
        # Delete the data
        redis_client.delete(full_key)
        
        return True
    except Exception as e:
        print(f"Error deleting from database: {e}")
        return False

def search_in_database(data_type: str, field: str, value: str):
    """Search data in database by field value"""
    if not redis_client:
        return []
    
    try:
        index_key = f"index:{data_type}"
        keys = redis_client.smembers(index_key)
        
        results = []
        for key in keys:
            full_key = f"{data_type}:{key}"
            data = redis_client.hgetall(full_key)
            
            if data and data.get(field) and value.lower() in data[field].lower():
                results.append({
                    'id': key,
                    'data': data
                })
        
        return results
    except Exception as e:
        print(f"Error searching in database: {e}")
        return []

def update_statistics(stat_name: str, increment: int = 1):
    """Update statistics counter"""
    if not redis_client:
        return
    
    try:
        redis_client.hincrby("system:statistics", stat_name, increment)
    except:
        pass

# ========== AI FUNCTIONS ==========
def generate_ai_response(user_message: str):
    """Generate AI response"""
    if not ai_client:
        # Mock response if AI is not available
        mock_responses = [
            "I understand your concern. Let me help you with that.",
            "That's a good question! Here's what I can tell you...",
            "I recommend checking our FAQ section for detailed information.",
            "For this issue, you might want to create a support ticket.",
            "I can assist you with that. Please provide more details."
        ]
        import random
        return random.choice(mock_responses)
    
    try:
        response = ai_client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant. Be friendly and professional."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"AI Error: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again."

# ========== DATABASE INITIALIZATION ==========
def initialize_database():
    """Initialize database with default data if empty"""
    if not redis_client:
        return
    
    # Default FAQ data
    faq_data = {
        "hours": "Our support is available 24/7. Live agents: 9 AM - 9 PM EST.",
        "location": "We are based in San Francisco with global support centers.",
        "return": "You can return products within 30 days of purchase.",
        "refund": "Refunds are processed within 5-7 business days.",
        "password": "Click 'Forgot Password' on login page or contact support.",
        "payment": "We accept Visa, MasterCard, PayPal, and bank transfers.",
        "shipping": "Standard shipping: 3-5 days, Express: 1-2 days.",
        "warranty": "All products come with 1-year manufacturer warranty.",
        "account": "You can update your account details in Profile section.",
        "technical": "For technical issues, please create a support ticket."
    }
    
    # Check if FAQ data exists
    if not redis_client.exists("index:faq_items"):
        print(" Initializing FAQ data in database...")
        
        # Save FAQ data to database
        for keyword, answer in faq_data.items():
            faq_item = {
                "question": keyword,
                "answer": answer,
                "category": "general",
                "created_at": datetime.now().isoformat()
            }
            save_to_database("faq_items", f"faq_{keyword}", faq_item)
        
        print(" FAQ data initialized in database")
    
    # Check if knowledge base exists
    if not redis_client.exists("index:knowledge_base"):
        print(" Initializing knowledge base in database...")
        
        # Default knowledge base items
        knowledge_items = [
            {
                "title": "Getting Started",
                "content": "Welcome to our support system. Here's how to get started with our platform...",
                "category": "general",
                "tags": ["getting started", "basics", "tutorial"]
            },
            {
                "title": "Troubleshooting Guide",
                "content": "Common troubleshooting steps for most issues: 1. Restart the application 2. Check your internet connection 3. Clear cache and cookies 4. Update to latest version",
                "category": "technical",
                "tags": ["troubleshooting", "technical", "guide"]
            },
            {
                "title": "Account Management",
                "content": "Learn how to manage your account settings, update profile information, and change security settings.",
                "category": "account",
                "tags": ["account", "profile", "settings"]
            }
        ]
        
        for item in knowledge_items:
            knowledge_id = f"knowledge_{item['category']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            item['id'] = knowledge_id
            save_to_database("knowledge_base", knowledge_id, item)
        
        print(" Knowledge base initialized in database")
    
    # Initialize statistics
    if not redis_client.exists("system:statistics"):
        print(" Initializing statistics...")
        redis_client.hset("system:statistics", "tickets_created", 0)
        redis_client.hset("system:statistics", "tickets_resolved", 0)
        redis_client.hset("system:statistics", "tickets_deleted", 0)
        redis_client.hset("system:statistics", "chats_processed", 0)

# Initialize database when server starts
initialize_database()

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "AI Customer Support Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Chat with AI",
            "/register": "POST - Register user",
            "/login": "POST - Login user",
            "/tickets": "GET/POST - Manage tickets",
            "/faq": "GET - Get FAQ",
            "/faq/items": "GET/POST - Manage FAQ items",
            "/knowledge": "GET/POST - Manage knowledge base",
            "/stats": "GET - Get statistics",
            "/search": "GET - Search database",
            "/backup": "GET - Backup all data",
            "/export/{type}": "GET - Export specific data"
        }
    }

@app.post("/register")
async def register(user: UserRegister):
    """Register new user"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Check if user exists
    if redis_client.exists(f"user:{user.email}"):
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": user.password,
        "created_at": datetime.now().isoformat()
    }
    
    create_user(user_data)
    
    # Create JWT token
    token = create_jwt_token(user.email)
    
    return {
        "message": "Registration successful",
        "token": token,
        "user": {
            "name": user.name,
            "email": user.email
        }
    }

@app.post("/login")
async def login(user: UserLogin):
    """Login user"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    user_data = authenticate_user(user.email, user.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create JWT token
    token = create_jwt_token(user.email)
    
    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "name": user_data.get("name"),
            "email": user.email
        }
    }

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    """Chat with AI assistant"""
    # First check FAQ from database
    response = None
    
    if redis_client:
        # Search in FAQ database
        faq_items = search_in_database("faq_items", "question", chat_request.message)
        faq_items.extend(search_in_database("faq_items", "answer", chat_request.message))
        
        if faq_items:
            response = faq_items[0]['data'].get('answer')
    
    # If no FAQ match, use AI
    if not response:
        response = generate_ai_response(chat_request.message)
    
    # Save chat history if Redis is available
    if redis_client:
        history_key = f"chat_history:{chat_request.user_id}"
        history_json = redis_client.get(history_key)
        history = json.loads(history_json) if history_json else []
        
        history.append({
            "timestamp": datetime.now().isoformat(),
            "user": chat_request.message,
            "bot": response
        })
        
        # Keep last 100 messages
        if len(history) > 100:
            history = history[-100:]
        
        redis_client.setex(history_key, 3600 * 24 * 7, json.dumps(history))  # 7 days expiry
        
        # Update statistics
        update_statistics("chats_processed", 1)
    
    return ChatResponse(response=response)

@app.post("/tickets")
async def create_ticket(ticket: Ticket):
    """Create a new support ticket"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Generate ticket ID
    ticket_id = f"ticket_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ticket.user_email[:5]}"
    
    # Prepare ticket data
    ticket_data = ticket.dict()
    ticket_data['id'] = ticket_id
    ticket_data['created_at'] = datetime.now().isoformat()
    ticket_data['updated_at'] = datetime.now().isoformat()
    
    # Save to database
    if save_to_database("tickets", ticket_id, ticket_data):
        # Update statistics
        update_statistics("tickets_created", 1)
        
        return {
            "message": "Ticket created successfully",
            "ticket_id": ticket_id,
            "ticket": ticket_data
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create ticket")

@app.get("/tickets")
async def get_tickets(user_email: Optional[str] = None, status: Optional[str] = None, 
                     priority: Optional[str] = None, category: Optional[str] = None):
    """Get all tickets with optional filters"""
    tickets_data = get_all_from_database("tickets")
    
    if not tickets_data:
        return {"tickets": []}
    
    tickets = []
    for item in tickets_data:
        ticket = item['data']
        
        # Apply filters
        if user_email and ticket.get('user_email') != user_email:
            continue
        if status and ticket.get('status') != status:
            continue
        if priority and ticket.get('priority') != priority:
            continue
        if category and ticket.get('category') != category:
            continue
        
        tickets.append(ticket)
    
    # Sort by creation date (newest first)
    tickets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return {"tickets": tickets}

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get a specific ticket by ID"""
    ticket_data = get_from_database("tickets", ticket_id)
    
    if not ticket_data:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    return {"ticket": ticket_data}

@app.put("/tickets/{ticket_id}")
async def update_ticket(ticket_id: str, update: TicketUpdate):
    """Update a ticket"""
    ticket_data = get_from_database("tickets", ticket_id)
    
    if not ticket_data:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Prepare update data
    update_data = update.dict(exclude_unset=True)
    
    if update_in_database("tickets", ticket_id, update_data):
        # Update statistics if status changed
        if 'status' in update_data and update_data['status'] == 'resolved':
            update_statistics("tickets_resolved", 1)
        
        return {"message": "Ticket updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update ticket")

@app.delete("/tickets/{ticket_id}")
async def delete_ticket(ticket_id: str):
    """Delete a ticket"""
    if delete_from_database("tickets", ticket_id):
        update_statistics("tickets_deleted", 1)
        return {"message": "Ticket deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Ticket not found")

@app.get("/faq")
async def get_faq():
    """Get all FAQ items"""
    faq_items = get_all_from_database("faq_items")
    
    # Convert to simple key-value format for backward compatibility
    faq_dict = {}
    for item in faq_items:
        question = item['data'].get('question', '')
        answer = item['data'].get('answer', '')
        if question and answer:
            faq_dict[question] = answer
    
    return {"faq": faq_dict}

@app.post("/faq/items")
async def create_faq_item(item: FAQItem):
    """Create a new FAQ item"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Generate FAQ ID
    faq_id = f"faq_{item.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare FAQ data
    faq_data = item.dict()
    faq_data['id'] = faq_id
    faq_data['created_at'] = datetime.now().isoformat()
    
    # Save to database
    if save_to_database("faq_items", faq_id, faq_data):
        return {
            "message": "FAQ item created successfully",
            "faq_id": faq_id,
            "item": faq_data
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create FAQ item")

@app.get("/faq/items")
async def get_faq_items(category: Optional[str] = None):
    """Get all FAQ items or filter by category"""
    faq_items_data = get_all_from_database("faq_items")
    
    if not faq_items_data:
        return {"faq_items": []}
    
    items = []
    for item in faq_items_data:
        faq_item = item['data']
        
        # Apply category filter
        if category and faq_item.get('category') != category:
            continue
        
        items.append(faq_item)
    
    return {"faq_items": items}

@app.post("/knowledge")
async def create_knowledge_item(item: KnowledgeBaseItem):
    """Create a new knowledge base item"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Generate knowledge ID
    knowledge_id = f"knowledge_{item.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare knowledge data
    knowledge_data = item.dict()
    knowledge_data['id'] = knowledge_id
    knowledge_data['created_at'] = datetime.now().isoformat()
    knowledge_data['views'] = 0
    
    # Save to database
    if save_to_database("knowledge_base", knowledge_id, knowledge_data):
        # Index tags
        for tag in item.tags:
            tag_key = f"tag:{tag}"
            redis_client.sadd(tag_key, knowledge_id)
        
        return {
            "message": "Knowledge base item created successfully",
            "knowledge_id": knowledge_id,
            "item": knowledge_data
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create knowledge base item")

@app.get("/knowledge")
async def get_knowledge_items(category: Optional[str] = None, tag: Optional[str] = None):
    """Get knowledge base items"""
    knowledge_items_data = get_all_from_database("knowledge_base")
    
    if not knowledge_items_data:
        return {"knowledge_items": []}
    
    items = []
    for item in knowledge_items_data:
        knowledge_item = item['data']
        
        # Apply category filter
        if category and knowledge_item.get('category') != category:
            continue
        
        # Apply tag filter
        if tag and tag not in knowledge_item.get('tags', []):
            continue
        
        items.append(knowledge_item)
    
    return {"knowledge_items": items}

@app.get("/chat/history")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user"""
    if not redis_client:
        return {"history": []}
    
    history_key = f"chat_history:{user_id}"
    history_json = redis_client.get(history_key)
    
    if history_json:
        history = json.loads(history_json)
        if limit:
            history = history[-limit:]
        return {"history": history}
    else:
        return {"history": []}

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get total users
    user_keys = redis_client.keys("user:*")
    total_users = len(user_keys)
    
    # Get tickets statistics
    tickets_data = get_all_from_database("tickets")
    total_tickets = len(tickets_data)
    open_tickets = 0
    resolved_tickets = 0
    
    for ticket in tickets_data:
        ticket_data = ticket['data']
        if ticket_data.get('status') == 'open':
            open_tickets += 1
        elif ticket_data.get('status') == 'resolved':
            resolved_tickets += 1
    
    # Get chat statistics
    chat_keys = redis_client.keys("chat_history:*")
    total_chats = len(chat_keys)
    
    # Get FAQ statistics
    faq_items = get_all_from_database("faq_items")
    total_faq_items = len(faq_items)
    
    # Get knowledge base statistics
    knowledge_items = get_all_from_database("knowledge_base")
    total_knowledge_items = len(knowledge_items)
    
    # Get system statistics
    system_stats = redis_client.hgetall("system:statistics")
    
    stats = {
        "total_users": total_users,
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "resolved_tickets": resolved_tickets,
        "total_chats": total_chats,
        "total_faq_items": total_faq_items,
        "total_knowledge_items": total_knowledge_items,
        "system_stats": system_stats,
        "uptime": datetime.now().isoformat()
    }
    
    return {"statistics": stats}

@app.get("/search")
async def search_all(query: str):
    """Search across all data types"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    results = {
        "tickets": search_in_database("tickets", "subject", query) + 
                  search_in_database("tickets", "description", query),
        "faq": search_in_database("faq_items", "question", query) + 
               search_in_database("faq_items", "answer", query),
        "knowledge": search_in_database("knowledge_base", "title", query) + 
                    search_in_database("knowledge_base", "content", query)
    }
    
    return {"query": query, "results": results}

@app.get("/export/{data_type}")
async def export_data(data_type: str):
    """Export data as JSON"""
    if data_type not in ["tickets", "faq_items", "knowledge_base", "users"]:
        raise HTTPException(status_code=400, detail="Invalid data type")
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    if data_type == "users":
        # Export users
        user_keys = redis_client.keys("user:*")
        users = []
        for key in user_keys:
            user_data = redis_client.hgetall(key)
            # Remove password for security
            if 'password' in user_data:
                del user_data['password']
            users.append({
                "key": key,
                "data": user_data
            })
        return {"data_type": data_type, "items": users}
    else:
        # Export other data types
        items = get_all_from_database(data_type)
        return {"data_type": data_type, "items": items}

@app.get("/backup")
async def create_backup():
    """Create a backup of all data"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "users": [],
        "tickets": get_all_from_database("tickets"),
        "faq_items": get_all_from_database("faq_items"),
        "knowledge_base": get_all_from_database("knowledge_base"),
        "chat_histories": [],
        "system_stats": redis_client.hgetall("system:statistics")
    }
    
    # Get user data (without passwords)
    user_keys = redis_client.keys("user:*")
    for key in user_keys:
        user_data = redis_client.hgetall(key)
        if 'password' in user_data:
            del user_data['password']
        backup_data["users"].append({
            "key": key,
            "data": user_data
        })
    
    # Get chat histories
    chat_keys = redis_client.keys("chat_history:*")
    for key in chat_keys:
        history_json = redis_client.get(key)
        if history_json:
            backup_data["chat_histories"].append({
                "key": key,
                "data": json.loads(history_json)
            })
    
    return backup_data

@app.post("/restore")
async def restore_backup(backup: dict):
    """Restore data from backup"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database not available")
    
    restored_count = 0
    
    # Restore users
    for user_item in backup.get("users", []):
        key = user_item["key"]
        data = user_item["data"]
        redis_client.hset(key, mapping=data)
        restored_count += 1
    
    # Restore tickets
    for ticket_item in backup.get("tickets", []):
        key = ticket_item["id"]
        data = ticket_item["data"]
        save_to_database("tickets", key, data)
        restored_count += 1
    
    # Restore FAQ items
    for faq_item in backup.get("faq_items", []):
        key = faq_item["id"]
        data = faq_item["data"]
        save_to_database("faq_items", key, data)
        restored_count += 1
    
    # Restore knowledge base
    for kb_item in backup.get("knowledge_base", []):
        key = kb_item["id"]
        data = kb_item["data"]
        save_to_database("knowledge_base", key, data)
        restored_count += 1
    
    # Restore chat histories
    for chat_item in backup.get("chat_histories", []):
        key = chat_item["key"]
        data = chat_item["data"]
        redis_client.set(key, json.dumps(data))
        restored_count += 1
    
    # Restore system statistics
    if backup.get("system_stats"):
        for stat, value in backup["system_stats"].items():
            redis_client.hset("system:statistics", stat, value)
    
    return {
        "message": f"Successfully restored {restored_count} items",
        "restored_count": restored_count
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_client and redis_client.ping() else "disconnected"
    ai_status = "available" if ai_client else "unavailable"
    
    return {
        "status": "ok",
        "redis": redis_status,
        "ai": ai_status,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8060,
        reload=True
    )