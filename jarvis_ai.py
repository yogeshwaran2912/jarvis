import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import threading
import time
import schedule
from pathlib import Path
import logging
from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_groq import ChatGroq
import pyaudio
import wave
import audioop
import subprocess
import platform
import sys
import serial
import re
import pyautogui
import pygetwindow as gw


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

word_to_number = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10
}

class MarauderController:
    def __init__(self, port='/dev/tty.usbserial-57670610721', baud=115200):
        print("[*] Connecting to ESP32 Marauder...")
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print("[+] Connected.")
        self.tts_engine = pyttsx3.init()

    def speak(self, text):
        print(f"[Voice] {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self, prompt="Say something..."):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self.speak(prompt)
            print("[*] Listening...")
            try:
                audio = r.listen(source, timeout=10)
                result = r.recognize_google(audio)
                print(f"[Recognized Voice Input]: {result}")
                return result.lower()
            except sr.UnknownValueError:
                print("[!] Could not understand audio.")
                self.speak("Sorry, I didn't understand.")
                return None
            except sr.RequestError:
                print("[!] Speech recognition service unavailable.")
                self.speak("Speech service unavailable.")
                return None
            except Exception as e:
                print(f"[!] Error: {e}")
                return None

    def send_cmd(self, cmd, wait=0.5):
        self.ser.write((cmd + '\n').encode())
        time.sleep(wait)

    def read_all(self, timeout=5):
        end_time = time.time() + timeout
        output = ""
        while time.time() < end_time:
            if self.ser.in_waiting:
                output += self.ser.read(self.ser.in_waiting).decode(errors="ignore")
            time.sleep(0.2)
        return output

    def scan_wifi(self):
        self.speak("Starting WiFi scan")
        self.send_cmd('scanap')
        time.sleep(15)
        self.send_cmd('stopscan')
        self.send_cmd('list -a')
        raw = self.read_all(timeout=5)

        print("\n[DEBUG RAW OUTPUT]\n" + "-"*30)
        print(raw)
        print("-"*30)

        ssids = re.findall(r'\[\d+\]\[CH:\d+\] (.+?) -\d+', raw)

        if not ssids:
            self.speak("No WiFi networks found.")
            return []

        self.speak("I found the following WiFi networks.")
        for i, ssid in enumerate(ssids):
            ssid_text = f"{i}: {ssid.strip()}"
            self.speak(ssid_text)

        return ssids

    def deauth(self, ssid_index):
        self.send_cmd(f'select -a {ssid_index}')
        time.sleep(1)
        self.send_cmd('attack -t deauth')
        self.speak(f"Deauth attack started on network index {ssid_index}")

    def stop_attack(self):
        self.send_cmd("stop")
        self.speak("Attack stopped.")

def parse_index_from_command(command):
    parts = command.split()
    for part in parts:
        if part in word_to_number:
            return word_to_number[part]
        elif part.isdigit():
            return int(part)
    return None

class JarvisAI:
    def __init__(self, gemini_api_key: str, groq_api_key: str, data_dir: str = "jarvis_data"):
        """
        Initialize Jarvis AI Assistant with Agentic capabilities
        
        Args:
            gemini_api_key: Google Gemini API key
            groq_api_key: Groq API key for agent processing
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize AI models
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        self.groq_model = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0)
        
        # Initialize TTS and STT
        self.tts_engine = pyttsx3.init()
        self.setup_voice()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Wake word detection
        self.wake_word = "jarvis"
        self.sleep_command = "jarvis go back to sleep"
        self.is_active = False
        self.listening_for_wake_word = True
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # File paths
        self.conversations_file = self.data_dir / "conversations.xlsx"
        self.tasks_file = self.data_dir / "tasks.xlsx"
        self.reminders_file = self.data_dir / "reminders.xlsx"
        self.vector_db_file = self.data_dir / "vector_db.pkl"
        self.faiss_index_file = self.data_dir / "faiss_index.idx"
        
        # Initialize data structures
        self.conversations = []
        self.tasks = []
        self.reminders = []
        self.vector_db = []
        self.faiss_index = None
        
        # Load existing data
        self.load_data()
        
        # Initialize vector database
        self.initialize_vector_db()
        
        # User context for better understanding
        self.user_context = {
            "preferences": {},
            "frequent_topics": {},
            "conversation_history": []
        }
        
        # Initialize agents and tools
        self.setup_agents()
        
        # Start background services
        self.start_reminder_scheduler()
        
        # Conversation memory for basic context (last 5 exchanges)
        self.conversation_chain = []
        
        # --- MarauderController and helpers ---
        self.marauder_controller = None  # Lazy init
        self.last_wifi_scan = []
        
        logger.info("Jarvis AI Assistant initialized successfully!")
    
    def setup_voice(self):
        """Configure voice settings"""
        voices = self.tts_engine.getProperty('voices')
        
        if voices:
            # Platform-specific voice selection for JARVIS-like sound
            system = platform.system()
            
            if system == "Windows":
                # Windows voices that sound more sophisticated
                preferred_voices = [
                    'david',           # Microsoft David (deeper, more professional)
                    'mark',            # Microsoft Mark (British accent)
                    'daniel',          # Daniel (UK English)
                    'male'             # Any male voice as fallback
                ]
            elif system == "Darwin":  # macOS
                # macOS voices with sophisticated sound
                preferred_voices = [
                    'daniel',          # Daniel (UK English) - very JARVIS-like
                    'alex',            # Alex (default male voice)
                    'tom',             # Tom (novelty voice)
                    'oliver',          # Oliver (UK English)
                    'male'
                ]
            else:  # Linux
                preferred_voices = [
                    'english-us',
                    'english-uk', 
                    'male'
                ]
            
            # Find and set the best available voice
            voice_set = False
            for preferred in preferred_voices:
                for voice in voices:
                    voice_name = voice.name.lower()
                    if preferred in voice_name:
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"Selected voice: {voice.name}")
                        voice_set = True
                        break
                if voice_set:
                    break
        
        # JARVIS-like speech parameters
        self.tts_engine.setProperty('rate', 180)      # Slightly slower, more deliberate
        self.tts_engine.setProperty('volume', 1.0)    # Clear and confident
        
        # Additional properties if supported
        try:
            # Some engines support pitch control
            self.tts_engine.setProperty('pitch', 0.7)  # Slightly lower pitch
        except:
            pass
    
    def setup_agents(self):
        """Setup LangChain agents and tools"""
        # Define tools for the agent
        self.tools = [
            Tool(
                name="add_task",
                func=self.agent_add_task,
                description="Add a new task. Input should be task description, optionally with due date and priority separated by commas."
            ),
            Tool(
                name="get_tasks",
                func=self.agent_get_tasks,
                description="Get all tasks or tasks with specific status. Input can be 'all', 'pending', 'completed', or 'in_progress'."
            ),
            Tool(
                name="list_tasks",
                func=self.agent_list_tasks,
                description="List all tasks with their details. Input should be empty."
            ),
            Tool(
                name="update_task",
                func=self.agent_update_task,
                description="Update task status. Input should be 'task_id,new_status' where status is 'pending', 'in_progress', or 'completed'."
            ),
            Tool(
                name="add_reminder",
                func=self.agent_add_reminder,
                description="Add a reminder. Input should be 'reminder_text,time' where time is in HH:MM format."
            ),
            Tool(
                name="get_reminders",
                func=self.agent_get_reminders,
                description="Get all active reminders."
            ),
            Tool(
                name="search_conversations",
                func=self.agent_search_conversations,
                description="Search previous conversations. Input should be the search query."
            ),
            Tool(
                name="get_user_context",
                func=self.agent_get_user_context,
                description="Get user context including preferences and frequent topics."
            ),
            Tool(
                name="bluetooth_flood",
                func=self.agent_flood_bt,
                description="Performs a Bluetooth flooding attack, which repeatedly sends discovery or connection requests to nearby Bluetooth devices, potentially overwhelming them. Use this to test Bluetooth resilience or simulate denial-of-service scenarios. No input is required, but a command like 'flood bluetooth' can be used to trigger it."
            )
        ]
        self.tools += [
            Tool(
                name="attack_wifi_network",
                func=self.agent_attack_wifi_network,
                description="Scan for WiFi networks and perform a wifi test. Input should be empty or a phrase like 'attack wifi'."
            ),
            Tool(
                name="stop_wifi_testing",
                func=self.agent_stop_wifi_testing,
                description="Stop the ongoing WiFi testing/attack. Input should be empty or a phrase like 'stop wifi testing'."
            ),
            Tool(
                name="flood_wifi",
                func=self.agent_flood_wifi,
                description="Send a WiFi beacon. Input should be empty or a phrase like 'flood wifi'."
            ),
            Tool(
                name="start_app",
                func=self.agent_start_app,
                description="Start an application. Input should be the application name like 'chrome', 'vscode', 'spotify', 'discord', 'terminal', etc."
            ),
            Tool(
                name="type_text",
                func=self.agent_type_text,
                description="Type text into the currently active application. Input should be the text to type. Use this after opening an app to write content into it."
            )
            
            
        ]
        
        # Load ReAct prompt
        self.prompt_react = hub.pull("hwchase17/react")
        
        # Create ReAct agent
        self.react_agent = create_react_agent(self.groq_model, tools=self.tools, prompt=self.prompt_react)
        self.agent_executor = AgentExecutor(
            agent=self.react_agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    # Agent Tool Functions
    def agent_add_task(self, input_string: str) -> str:
        """Agent tool to add tasks"""
        parts = input_string.split(',')
        task_desc = parts[0].strip()
        due_date = parts[1].strip() if len(parts) > 1 else None
        priority = parts[2].strip() if len(parts) > 2 else "medium"
        return self.add_task(task_desc, due_date, priority)
    
    def agent_get_tasks(self, status: str = "all") -> str:
        """Agent tool to get tasks"""
        if status == "all":
            tasks = self.get_tasks()
        else:
            tasks = self.get_tasks(status.strip())
        
        if not tasks:
            return f"No tasks found with status: {status}"
        
        task_list = []
        for task in tasks:
            task_list.append(f"ID: {task['id']} - {task['description']} (Status: {task['status']}, Priority: {task['priority']})")
        
        return "\n".join(task_list)
    
    def agent_list_tasks(self, input_string: str = "") -> str:
        """Agent tool to list all tasks with details"""
        tasks = self.get_tasks()
        if not tasks:
            return "No tasks found."
        task_list = []
        for task in tasks:
            task_list.append(
                f"ID: {task['id']} - {task['description']} (Status: {task['status']}, Due: {task.get('due_date','')}, Priority: {task.get('priority','')})"
            )
        return "\n".join(task_list)

    
    def agent_update_task(self, input_string: str) -> str:
        """Agent tool to update task status"""
        parts = input_string.split(',')
        if len(parts) != 2:
            return "Invalid input. Please provide task_id,new_status"
        
        try:
            task_id = int(parts[0].strip())
            new_status = parts[1].strip()
            return self.update_task_status(task_id, new_status)
        except ValueError:
            return "Invalid task ID. Please provide a valid number."
    
    def agent_add_reminder(self, input_string: str) -> str:
        """Agent tool to add reminders"""
        parts = input_string.split(',')
        if len(parts) != 2:
            return "Invalid input. Please provide reminder_text,time"
        
        reminder_text = parts[0].strip()
        reminder_time = parts[1].strip()
        return self.add_reminder(reminder_text, reminder_time)
    
    def agent_get_reminders(self, input_string: str = "") -> str:
        """Agent tool to get reminders"""
        active_reminders = [r for r in self.reminders if r['status'] == 'active']
        if not active_reminders:
            return "No active reminders found."
        
        reminder_list = []
        for reminder in active_reminders:
            reminder_list.append(f"ID: {reminder['id']} - {reminder['text']} at {reminder['time']}")
        
        return "\n".join(reminder_list)
    
    def agent_search_conversations(self, query: str) -> str:
        """Agent tool to search conversations"""
        results = self.search_vector_db(query, k=3)
        if not results:
            return "No relevant conversations found."
        
        context = "Previous conversations:\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['text'][:200]}...\n"
        
        return context
    
    def agent_stop_wifi_testing(self, input_string: str = "") -> str:
        """Stop any ongoing WiFi testing/attacks"""
        if not self._ensure_marauder():
            return "Unable to connect to Marauder device."
            
        try:
            self.marauder_controller.stop_attack()
            return "WiFi testing/attack has been stopped."
        except Exception as e:
            logger.error(f"Error stopping WiFi test: {e}")
            return "Failed to stop WiFi testing due to an error."

    def agent_flood_wifi(self, input_string: str = "") -> str:
        """Execute a WiFi beacon flood attack"""
        if not self._ensure_marauder():
            return "Unable to connect to Marauder device."
        try:
            self.speak("Starting WiFi beacon flood attack")
            self.marauder_controller.send_cmd("attack -t beacon -r 10")
            return "WiFi beacon flood attack started. Use stop_wifi_testing to stop the attack."
        except Exception as e:
            logger.error(f"Error starting beacon flood: {e}")
        return "Failed to start WiFi beacon flood attack due to an error."
    
    def agent_flood_bt(self, input_string: str = "") -> str:
        """Execute a WiFi beacon flood attack"""
        print("Starting Bluetooth flood attack")
        if not self._ensure_marauder():
            return "Unable to connect to Marauder device."
        try:
            self.speak("Starting bluetooth spam attack")
            self.marauder_controller.send_cmd("blespam -t all")
            return "bluetooth spam attack started. Use stop_wifi_testing to stop the attack."
        except Exception as e:
            logger.error(f"Error starting beacon flood: {e}")
        return "Failed to start bluetooth spam attack due to an error."

    
    def agent_get_user_context(self, input_string: str = "") -> str:
        """Agent tool to get user context"""
        frequent_topics = sorted(self.user_context['frequent_topics'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        context = f"User preferences: {self.user_context['preferences']}\n"
        context += f"Top topics: {[topic[0] for topic in frequent_topics]}\n"
        context += f"Recent conversations: {len(self.user_context['conversation_history'])}"
        
        return context
    
    # --- Combined WiFi attack tool ---
    def agent_attack_wifi_network(self, input_string: str = "") -> str:
        if not self._ensure_marauder():
            return "Unable to connect to Marauder device."
        ssids = self.marauder_controller.scan_wifi()
        if not ssids:
            return "No WiFi networks found."
        # Speak and print the list
        for i, ssid in enumerate(ssids):
            self.speak(f"{i}: {ssid.strip()}")
        self.speak("Please say the index of the WiFi network to attack.")
        # Listen for index
        idx = None
        for _ in range(3):
            index_cmd = self.marauder_controller.listen("Say the index number.")
            if index_cmd:
                idx = parse_index_from_command(index_cmd)
                if idx is not None and 0 <= idx < len(ssids):
                    break
                else:
                    self.speak("Invalid index. Please try again.")
            else:
                self.speak("No input received. Please try again.")
        if idx is None or idx < 0 or idx >= len(ssids):
            return "Failed to get a valid WiFi index. Aborting attack."
        self.marauder_controller.deauth(idx)
        return f"Deauth attack started on WiFi network {idx}: {ssids[idx].strip()}"

    def agent_start_app(self, app_name: str) -> str:
        """Agent tool to start applications"""
        try:
            app_name = app_name.strip().lower()
            
            # Common application mappings
            app_commands = {
                'chrome': 'open -a "Google Chrome"' if platform.system() == 'Darwin' else 'start chrome',
                'firefox': 'open -a "Firefox"' if platform.system() == 'Darwin' else 'start firefox',
                'safari': 'open -a "Safari"' if platform.system() == 'Darwin' else 'start safari',
                'vscode': 'open -a "Visual Studio Code"' if platform.system() == 'Darwin' else 'start code',
                'spotify': 'open -a "Spotify"' if platform.system() == 'Darwin' else 'start spotify',
                'discord': 'open -a "Discord"' if platform.system() == 'Darwin' else 'start discord',
                'terminal': 'open -a "Terminal"' if platform.system() == 'Darwin' else 'start cmd',
                'calculator': 'open -a "Calculator"' if platform.system() == 'Darwin' else 'start calc',
                'notes': 'open -a "Notes"' if platform.system() == 'Darwin' else 'start notepad',
                'finder': 'open -a "Finder"' if platform.system() == 'Darwin' else 'start explorer',
                'music': 'open -a "Music"' if platform.system() == 'Darwin' else 'start wmplayer',
                'mail': 'open -a "Mail"' if platform.system() == 'Darwin' else 'start outlook',
                'slack': 'open -a "Slack"' if platform.system() == 'Darwin' else 'start slack',
                'zoom': 'open -a "Zoom"' if platform.system() == 'Darwin' else 'start zoom',
                'teams': 'open -a "Microsoft Teams"' if platform.system() == 'Darwin' else 'start teams'
            }
            
            # Check if app is in our known mappings
            if app_name in app_commands:
                command = app_commands[app_name]
                self.speak(f"Starting {app_name}")
                try:
                    # Use subprocess.run for better reliability
                    if platform.system() == 'Darwin':
                        # For macOS, split the command properly
                        subprocess.run(['open', '-a', app_name.title()], check=True)
                    else:
                        subprocess.run(command, shell=True, check=True)
                    return f"Successfully started {app_name}"
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to start {app_name}: {e}")
                    return f"Failed to start {app_name}. The application may not be installed."
            else:
                # Try to open with system default
                try:
                    if platform.system() == 'Darwin':
                        subprocess.run(['open', '-a', app_name.title()], check=True)
                    elif platform.system() == 'Windows':
                        subprocess.run(['start', app_name], shell=True, check=True)
                    else:  # Linux
                        subprocess.run([app_name], check=True)
                    
                    self.speak(f"Started {app_name}")
                    return f"Successfully started {app_name}"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return f"Could not start {app_name}. Please check if it's installed or try a different name."
                
        except FileNotFoundError:
            return f"Application '{app_name}' not found. Please check if it's installed or try a different name."
        except Exception as e:
            logger.error(f"Error starting app {app_name}: {e}")
            return f"Failed to start {app_name}: {str(e)}"

    def agent_type_text(self, text_to_type: str) -> str:
        """Agent tool to type text into the currently active application"""
        try:
            if not text_to_type or not text_to_type.strip():
                return "No text provided to type."
            
            text_to_type = text_to_type.strip()
            
            # Safety check - limit text length to prevent accidental large pastes
            if len(text_to_type) > 5000:
                return "Text too long. Please provide text under 5000 characters."
            
            self.speak(f"Typing text: {text_to_type[:50]}{'...' if len(text_to_type) > 50 else ''}")
            
            # Add a delay to ensure the target application is ready and focused
            time.sleep(2)
            
            # Try to ensure we have a text input field focused
            # Click in the center of the screen to focus on the main window
            screen_width, screen_height = pyautogui.size()
            pyautogui.click(screen_width/2, screen_height/2)
            time.sleep(0.5)
            
            # Type the text using pyautogui with a reasonable interval
            pyautogui.typewrite(text_to_type, interval=0.02)  # Slightly longer interval for reliability
            
            return f"Successfully typed {len(text_to_type)} characters into the active application."
            
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return f"Failed to type text: {str(e)}"

    def _ensure_marauder(self):
        if self.marauder_controller is None:
            try:
                self.marauder_controller = MarauderController()
            except Exception as e:
                logger.error(f"Could not initialize MarauderController: {e}")
                return False
        return True

    def load_data(self):
        """Load existing data from Excel file"""
        try:
            # Load conversations
            if self.conversations_file.exists():
                df = pd.read_excel(self.conversations_file)
                self.conversations = df.to_dict('records')
            
            # Load reminders
            if self.reminders_file.exists():
                df = pd.read_excel(self.reminders_file)
                self.reminders = df.to_dict('records')
            
            # Load tasks from Excel file
            if self.tasks_file.exists():
                df = pd.read_excel(self.tasks_file)
                self.tasks = df.to_dict('records')
            
            # Load vector database
            if self.vector_db_file.exists():
                with open(self.vector_db_file, 'rb') as f:
                    self.vector_db = pickle.load(f)
            
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def save_data(self):
        """Save data to Excel file"""
        try:
            # Save conversations
            if self.conversations:
                df = pd.DataFrame(self.conversations)
                df.to_excel(self.conversations_file, index=False)
            
            # Save reminders
            if self.reminders:
                df = pd.DataFrame(self.reminders)
                df.to_excel(self.reminders_file, index=False)
            
            # Save tasks to Excel file
            if self.tasks:
                df = pd.DataFrame(self.tasks)
                df.to_excel(self.tasks_file, index=False)
            
            # Save vector database
            with open(self.vector_db_file, 'wb') as f:
                pickle.dump(self.vector_db, f)
            
            logger.info("Data saved successfully!")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def initialize_vector_db(self):
        """Initialize FAISS vector database"""
        try:
            if self.vector_db and len(self.vector_db) > 0:
                embeddings = np.array([item['embedding'] for item in self.vector_db])
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Save FAISS index
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            else:
                # Create empty index
                dimension = 384  # all-MiniLM-L6-v2 dimension
                self.faiss_index = faiss.IndexFlatL2(dimension)
            
            logger.info("Vector database initialized!")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            dimension = 384
            self.faiss_index = faiss.IndexFlatL2(dimension)
    
    def speak(self, text: str):
        """Convert text to speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
    
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Convert speech to text with improved settings"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            text = self.recognizer.recognize_google(audio)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return None
    
    def detect_wake_word(self) -> bool:
        """Detect wake word in continuous listening mode"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            text = self.recognizer.recognize_google(audio).lower().strip()
            print(text)
            return self.wake_word in text
        except:
            return False
    
    def add_to_vector_db(self, text: str, metadata: Dict[str, Any]):
        """Add text and metadata to vector database"""
        try:
            embedding = self.sentence_model.encode(text)
            
            vector_item = {
                'text': text,
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.vector_db.append(vector_item)
            
            # Add to FAISS index
            if self.faiss_index is not None:
                self.faiss_index.add(np.array([embedding]).astype('float32'))
            
            logger.info("Added to vector database")
        except Exception as e:
            logger.error(f"Error adding to vector database: {e}")
    
    def search_vector_db(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for similar conversations"""
        try:
            if not self.vector_db or self.faiss_index is None:
                return []
            
            query_embedding = self.sentence_model.encode(query)
            distances, indices = self.faiss_index.search(
                np.array([query_embedding]).astype('float32'), k
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.vector_db):
                    result = self.vector_db[idx].copy()
                    result['similarity_score'] = 1 / (1 + distances[0][i])
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def should_use_agent(self, user_input: str) -> bool:
        """Determine if agent should handle the request"""
        agent_keywords = [
            'task', 'reminder', 'schedule', 'add', 'complete', 'finish',
            'what did', 'previous', 'before', 'earlier', 'context',
            'search', 'find', 'look for', 'remember',
            'wifi', 'wi-fi', 'scan network', 'deauth', 'attack', 'stop attack'
        ]
        
        user_lower = user_input.lower()
        # Only trigger the new tool for attack/deauth/scan wifi
        return any(keyword in user_lower for keyword in agent_keywords)

    def process_with_agent(self, user_input: str) -> str:
        """Process user input using the agent"""
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result.get("output", "I encountered an error processing your request.")
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            return "I encountered an error while processing your request. Let me try a different approach."
    
    def generate_response_with_context(self, user_input: str) -> str:
        """Generate response using Gemini with context"""
        try:
            # --- PATCH: Add recent conversation chain for follow-up context ---
            chain_context = ""
            if self.conversation_chain:
                chain_context = "Recent conversation:\n"
                for turn in self.conversation_chain[-5:]:
                    chain_context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            
            # Get context from previous conversations
            similar_conversations = self.search_vector_db(user_input, k=3)
            
            context = ""
            if similar_conversations:
                context = "Previous relevant conversations:\n"
                for conv in similar_conversations:
                    context += f"- {conv['text'][:150]}...\n"
            
            # Add user preferences
            if self.user_context['preferences']:
                context += f"\nUser preferences: {self.user_context['preferences']}\n"
            
            # --- Combine all context ---
            prompt = f"""
            You are Jarvis, a helpful AI assistant. Use the following context to provide personalized responses.

            {chain_context}
            {context}

            Current user input: {user_input}

            Provide a helpful, conversational response. Keep it concise but informative.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request."
    
    def process_user_input(self, user_input: str) -> str:
        """Main method to process user input"""
        # Check if user wants to sleep
        if self.sleep_command in user_input.lower():
            self.is_active = False
            self.listening_for_wake_word = True
            return "Going back to sleep mode. Say 'Jarvis' to wake me up."
        
        # Determine processing method
        if self.should_use_agent(user_input):
            response = self.process_with_agent(user_input)
        else:
            response = self.generate_response_with_context(user_input)
        
        # --- PATCH: Update conversation chain for follow-up context ---
        self._update_conversation_chain(user_input, response)
        
        # Save conversation
        self.save_conversation(user_input, response)
        
        return response
    
    def _update_conversation_chain(self, user_input: str, ai_response: str):
        """Maintain a short conversation chain for follow-up context"""
        self.conversation_chain.append({'user': user_input, 'ai': ai_response})
        # Keep only the last 5 exchanges
        if len(self.conversation_chain) > 5:
            self.conversation_chain = self.conversation_chain[-5:]

    def save_conversation(self, user_input: str, ai_response: str):
        """Save conversation to Excel and vector database"""
        conversation = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'conversation_id': len(self.conversations) + 1
        }
        
        self.conversations.append(conversation)
        
        # Add to vector database
        full_text = f"User: {user_input} | AI: {ai_response}"
        metadata = {
            'type': 'conversation',
            'timestamp': conversation['timestamp'],
            'conversation_id': conversation['conversation_id']
        }
        self.add_to_vector_db(full_text, metadata)
        
        # Update user context
        self.update_user_context(user_input)
        
        # Save to Excel
        self.save_data()
    
    def update_user_context(self, user_input: str):
        """Update user context based on conversation"""
        # Add to conversation history
        self.user_context['conversation_history'].append({
            'input': user_input,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Keep only last 50 conversations in memory
        if len(self.user_context['conversation_history']) > 50:
            self.user_context['conversation_history'] = self.user_context['conversation_history'][-50:]
        
        # Extract and update frequent topics
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                self.user_context['frequent_topics'][word] = self.user_context['frequent_topics'].get(word, 0) + 1
    
    def add_task(self, task_description: str, due_date: str = None, priority: str = "medium"):
        """Add a new task with id, status, etc."""
        next_id = max([t['id'] for t in self.tasks], default=0) + 1
        # PATCH: Store due_date as string if present, else empty string
        task = {
            'id': next_id,
            'description': task_description,
            'due_date': due_date if due_date else "",
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.datetime.now().isoformat()
        }
        self.tasks.append(task)
        self.save_data()
        return f"Task added: {task_description}{', ' + due_date if due_date else ''}"

    def get_tasks(self, status: str = None):
        """Return all tasks as a list of dicts, optionally filtered by status"""
        # PATCH: Always return a list of dicts, never None or string
        if not isinstance(self.tasks, list):
            self.tasks = []
        if status:
            return [task for task in self.tasks if task.get('status') == status]
        return list(self.tasks)

    def update_task_status(self, task_id: int, new_status: str):
        """Update task status by id"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['status'] = new_status
                task['updated_at'] = datetime.datetime.now().isoformat()
                self.save_data()
                return f"Task {task_id} status updated to {new_status}"
        return f"Task {task_id} not found"
    
    def add_reminder(self, reminder_text: str, reminder_time: str):
        """Add a new reminder"""
        reminder = {
            'id': len(self.reminders) + 1,
            'text': reminder_text,
            'time': reminder_time,
            'status': 'active',
            'created_at': datetime.datetime.now().isoformat()
        }
        
        self.reminders.append(reminder)
        self.save_data()
        
        return f"Reminder set: {reminder_text} at {reminder_time}"
    
    def check_reminders(self):
        """Check and trigger reminders"""
        current_time = datetime.datetime.now().strftime("%H:%M")
        
        for reminder in self.reminders:
            if (reminder['status'] == 'active' and 
                reminder['time'] == current_time):
                
                reminder_msg = f"REMINDER: {reminder['text']}"
                print(reminder_msg)
                self.speak(f"Reminder: {reminder['text']}")
                
                # Mark reminder as completed
                reminder['status'] = 'completed'
                self.save_data()
    
    def start_reminder_scheduler(self):
        """Start the reminder scheduler in a separate thread"""
        def run_scheduler():
            schedule.every().minute.do(self.check_reminders)
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def run_continuous_mode(self):
        """Run Jarvis in continuous voice mode with wake word detection"""
        print("Jarvis is now running in continuous mode...")
        print(f"Say '{self.wake_word}' to activate")
        print(f"Say '{self.sleep_command}' to deactivate")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                if self.listening_for_wake_word:
                    # Listen for wake word
                    if self.detect_wake_word():
                        print("Wake word detected!")
                        self.speak("Yes, I'm listening. How can I help you?")
                        self.is_active = True
                        self.listening_for_wake_word = False
                
                elif self.is_active:
                    # Active conversation mode
                    print("Listening... (Say something or wait for timeout)")
                    user_input = self.listen(timeout=10, phrase_time_limit=15)
                    
                    if user_input:
                        print(f"You said: {user_input}")
                        
                        # Process the input
                        response = self.process_user_input(user_input)
                        
                        print(f"Jarvis: {response}")
                        self.speak(response)
                        
                        # Check if going back to sleep
                        if not self.is_active:
                            print("Jarvis is now sleeping. Say 'Jarvis' to wake up.")
                    else:
                        # Timeout - ask if user is still there
                        print("No input detected. Going back to sleep mode.")
                        self.speak("I'll go back to sleep now. Say Jarvis to wake me up.")
                        self.is_active = False
                        self.listening_for_wake_word = True
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\nJarvis shutting down...")
            self.speak("Goodbye!")
    
    def run_interactive_mode(self):
        """Run Jarvis in text-based interactive mode"""
        print("Jarvis Interactive Mode - Type 'exit' to quit")
        print("Jarvis: Hello! I'm Jarvis, your AI assistant. How can I help you today?")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'goodbye']:
                print("Jarvis: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Process the input
            response = self.process_user_input(user_input)
            print(f"Jarvis: {response}")

# Usage example
if __name__ == "__main__":
    # Initialize Jarvis with your API keys
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Get from environment variable
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")      # Get from environment variable

    if not GEMINI_API_KEY or not GROQ_API_KEY:
        print("Error: Please set GEMINI_API_KEY and GROQ_API_KEY environment variables.")
        sys.exit(1)
    
    try:
        jarvis = JarvisAI(GEMINI_API_KEY, GROQ_API_KEY)
        
        # Choose mode
        print("Choose interaction mode:")
        print("1. Continuous Voice Mode (with wake word detection)")
        print("2. Interactive Text Mode")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            jarvis.run_continuous_mode()
        else:
            jarvis.run_interactive_mode()
            
    except Exception as e:
        print(f"Error initializing Jarvis: {e}")
        print("Please make sure you have set your API keys correctly.")
