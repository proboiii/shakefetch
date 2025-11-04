import sys
import os
import hvsrpy
from hvsrpy import sesame

# Determine the base directory, whether running as a script or as a frozen exe
# if getattr(sys, 'frozen', False):
#     # If the application is run as a bundle, the base directory is the directory of the executable
#     base_dir = os.path.dirname(sys.executable)
# else:
#     # If run as a normal script, it's the script's directory
#     base_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the local obspy library
# obspy_path = os.path.join(base_dir, 'obspy-master')

# # Add the obspy path to the system path, so that the bundled obspy library can be imported.
# if os.path.isdir(obspy_path):
#     sys.path.insert(0, obspy_path)


import json
import keyring
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from ttkthemes import ThemedTk
from obspy import UTCDateTime
import obspy.io.mseed
import threading
import queue
import logging
from datetime import datetime, timedelta, timezone
import numpy as np

# Import logic
from time_sync import ShakeCommunicator
from data_acquisition import fetch_waveforms
from mhvsr_logic import process_mhvsr, get_default_preprocessing_settings, get_default_processing_settings

PROFILES_FILE = "profiles.json"
KEYRING_SERVICE = "ShakeFetch"

class DateTimePicker(tk.Toplevel):
    def __init__(self, parent, entry_widget):
        super().__init__(parent)
        self.entry_widget = entry_widget
        self.title("Select Date and Time")

        now = datetime.now(timezone.utc)
        try:
            dt_str = self.entry_widget.get()
            now = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass

        self.year = tk.IntVar(value=now.year)
        self.month = tk.IntVar(value=now.month)
        self.day = tk.IntVar(value=now.day)
        self.hour = tk.IntVar(value=now.hour)
        self.minute = tk.IntVar(value=now.minute)
        self.second = tk.IntVar(value=now.second)

        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Year:").grid(row=0, column=0)
        ttk.Spinbox(frame, from_=1970, to=2100, textvariable=self.year, width=5).grid(row=0, column=1)
        ttk.Label(frame, text="Month:").grid(row=0, column=2)
        ttk.Spinbox(frame, from_=1, to=12, textvariable=self.month, width=3).grid(row=0, column=3)
        ttk.Label(frame, text="Day:").grid(row=0, column=4)
        ttk.Spinbox(frame, from_=1, to=31, textvariable=self.day, width=3).grid(row=0, column=5)

        ttk.Label(frame, text="Hour:").grid(row=1, column=0)
        ttk.Spinbox(frame, from_=0, to=23, textvariable=self.hour, width=3).grid(row=1, column=1)
        ttk.Label(frame, text="Minute:").grid(row=1, column=2)
        ttk.Spinbox(frame, from_=0, to=59, textvariable=self.minute, width=3).grid(row=1, column=3)
        ttk.Label(frame, text="Second:").grid(row=1, column=4)
        ttk.Spinbox(frame, from_=0, to=59, textvariable=self.second, width=3).grid(row=1, column=5)

        ttk.Button(self, text="Done", command=self.on_done).pack(pady=5)

    def on_done(self):
        dt_str = f"{self.year.get():04d}-{self.month.get():02d}-{self.day.get():02d}T{self.hour.get():02d}:{self.minute.get():02d}:{self.second.get():02d}"
        self.entry_widget.delete(0, tk.END)
        self.entry_widget.insert(0, dt_str)
        self.destroy()

class ShakeFetchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ShakeFetch")
        self.root.geometry("850x750") # Increased height for profile UI
        self.shake_communicator = None
        self.profiles = {}
        self.remember_ssh_var = tk.BooleanVar(value=True)

        # Setup logging
        self.setup_logging()

        # Style
        style = ttk.Style()
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        style.configure("TEntry", padding=5)
        style.configure("TLabelframe.Label", padding=5)

        # Queue for thread communication
        self.task_queue = queue.Queue()

        # Create a Notebook widget (for tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Create the tabs
        self.time_sync_tab = ttk.Frame(self.notebook)
        self.data_acquisition_tab = ttk.Frame(self.notebook)
        self.multifetch_tab = ttk.Frame(self.notebook)
        self.mhvsr_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.time_sync_tab, text="Shake Connection")
        self.notebook.add(self.data_acquisition_tab, text="Single Fetch")
        self.notebook.add(self.multifetch_tab, text="Multifetch")
        self.notebook.add(self.mhvsr_tab, text="MHVSR Analysis")

        # Populate the tabs
        self.create_time_sync_tab()
        self.create_data_acquisition_tab()
        self.create_multifetch_tab()
        self.create_mhvsr_tab()

        # Load profiles
        self.load_profiles()

        # Start the queue processor
        self.root.after(100, self.process_queue)
        logging.info("ShakeFetch application started.")

    def setup_logging(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        logging.basicConfig(filename="logs/shakefetch.log",
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

    def process_queue(self):
        try:
            message = self.task_queue.get_nowait()
            message[0](*message[1:])
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def start_task(self, worker_func, *args):
        thread = threading.Thread(target=worker_func, args=args)
        thread.daemon = True
        thread.start()

    # --- Profile Management ---
    def load_profiles(self):
        try:
            if os.path.exists(PROFILES_FILE):
                with open(PROFILES_FILE, 'r') as f:
                    self.profiles = json.load(f)
                self.profile_selector['values'] = list(self.profiles.keys())
                logging.info(f"Loaded {len(self.profiles)} profiles from {PROFILES_FILE}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading profiles: {e}")
            self.profiles = {}

    def save_profiles_to_file(self):
        try:
            with open(PROFILES_FILE, 'w') as f:
                json.dump(self.profiles, f, indent=4)
            logging.info(f"Saved profiles to {PROFILES_FILE}")
        except Exception as e:
            logging.error(f"Error saving profiles: {e}")
            messagebox.showerror("Profile Error", f"Could not save profiles to file: {e}")

    def on_profile_select(self, event):
        profile_name = self.profile_selector.get()
        if profile_name in self.profiles:
            profile_data = self.profiles[profile_name]
            self.update_all_fields(profile_data)
            
            # Retrieve password from keyring
            password = keyring.get_password(KEYRING_SERVICE, profile_name)
            if password:
                self.ts_password_entry.delete(0, tk.END)
                self.ts_password_entry.insert(0, password)
            logging.info(f"Loaded profile: {profile_name}")

    def update_all_fields(self, data):
        # Helper to update an entry
        def _update_entry(entry, value):
            entry.delete(0, tk.END)
            entry.insert(0, value)

        # Shake Connection Tab
        _update_entry(self.ts_host_entry, data.get("ts_host", "rs.local"))
        _update_entry(self.ts_username_entry, data.get("ts_username", "myshake"))

        # Single Fetch Tab
        _update_entry(self.da_host_entry, data.get("da_host", "rs.local"))
        _update_entry(self.da_port_entry, data.get("da_port", "16032"))
        _update_entry(self.da_net_entry, data.get("da_net", "AM"))
        _update_entry(self.da_sta_entry, data.get("da_sta", "R1E3F"))
        _update_entry(self.da_loc_entry, data.get("da_loc", "00"))
        _update_entry(self.da_cha_entry, data.get("da_cha", "EH*"))

        # Multifetch Tab
        _update_entry(self.mf_host_entry, data.get("mf_host", "rs.local"))
        _update_entry(self.mf_port_entry, data.get("mf_port", "16032"))
        _update_entry(self.mf_net_entry, data.get("mf_net", "AM"))
        _update_entry(self.mf_sta_entry, data.get("mf_sta", "R1E3F"))
        _update_entry(self.mf_loc_entry, data.get("mf_loc", "00"))
        _update_entry(self.mf_cha_entry, data.get("mf_cha", "EH*"))

    def save_profile(self):
        profile_name = self.profile_name_entry.get()
        if not profile_name:
            messagebox.showerror("Input Error", "Profile Name cannot be empty.")
            return

        profile_data = {
            "ts_host": self.ts_host_entry.get(),
            "ts_username": self.ts_username_entry.get(),
            "da_host": self.da_host_entry.get(),
            "da_port": self.da_port_entry.get(),
            "da_net": self.da_net_entry.get(),
            "da_sta": self.da_sta_entry.get(),
            "da_loc": self.da_loc_entry.get(),
            "da_cha": self.da_cha_entry.get(),
            "mf_host": self.mf_host_entry.get(),
            "mf_port": self.mf_port_entry.get(),
            "mf_net": self.mf_net_entry.get(),
            "mf_sta": self.mf_sta_entry.get(),
            "mf_loc": self.mf_loc_entry.get(),
            "mf_cha": self.mf_cha_entry.get(),
        }
        self.profiles[profile_name] = profile_data

        if self.remember_ssh_var.get():
            password = self.ts_password_entry.get()
            if password:
                keyring.set_password(KEYRING_SERVICE, profile_name, password)
                logging.info(f"Saved password for profile '{profile_name}' to keyring.")
        else:
            # If user unchecks, delete existing password
            try:
                keyring.delete_password(KEYRING_SERVICE, profile_name)
                logging.info(f"Deleted password for profile '{profile_name}' from keyring.")
            except keyring.errors.PasswordDeleteError:
                logging.warning(f"No password found for profile '{profile_name}' to delete.")

        self.save_profiles_to_file()
        self.profile_selector['values'] = list(self.profiles.keys())
        self.profile_selector.set(profile_name)
        messagebox.showinfo("Success", f"Profile '{profile_name}' saved successfully.")

    def delete_profile(self):
        profile_name = self.profile_selector.get()
        if not profile_name:
            messagebox.showerror("Selection Error", "No profile selected to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the profile '{profile_name}'?"):
            if profile_name in self.profiles:
                del self.profiles[profile_name]
                try:
                    keyring.delete_password(KEYRING_SERVICE, profile_name)
                    logging.info(f"Deleted password for profile '{profile_name}' from keyring.")
                except keyring.errors.PasswordDeleteError:
                    pass # No password was stored, which is fine.
                
                self.save_profiles_to_file()
                self.profile_selector['values'] = list(self.profiles.keys())
                self.profile_selector.set('')
                self.profile_name_entry.delete(0, tk.END)
                messagebox.showinfo("Success", f"Profile '{profile_name}' deleted.")

    # --- Shake Connection Tab ---
    def create_time_sync_tab(self):
        # --- Profile Management Frame ---
        profile_frame = ttk.LabelFrame(self.time_sync_tab, text="Profile Management", padding=(10, 5))
        profile_frame.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Label(profile_frame, text="Select Profile:").grid(row=0, column=0, sticky="w", pady=2)
        self.profile_selector = ttk.Combobox(profile_frame, state="readonly")
        self.profile_selector.grid(row=0, column=1, sticky="ew", padx=5)
        self.profile_selector.bind("<<ComboboxSelected>>", self.on_profile_select)

        ttk.Label(profile_frame, text="New Profile Name:").grid(row=1, column=0, sticky="w", pady=2)
        self.profile_name_entry = ttk.Entry(profile_frame)
        self.profile_name_entry.grid(row=1, column=1, sticky="ew", padx=5)

        # --- Controls Frame ---
        controls_frame = ttk.Frame(profile_frame)
        controls_frame.grid(row=0, column=2, rowspan=2, sticky="n", padx=5)

        ttk.Button(controls_frame, text="Save Profile", command=self.save_profile).pack(fill="x", pady=2)
        ttk.Button(controls_frame, text="Delete Selected", command=self.delete_profile).pack(fill="x", pady=2)
        
        self.remember_ssh_check = ttk.Checkbutton(controls_frame, text="Remember SSH Credentials", variable=self.remember_ssh_var)
        self.remember_ssh_check.pack(pady=(5,0))
        
        profile_frame.columnconfigure(1, weight=1)

        # --- Connection Details Frame ---
        input_frame = ttk.LabelFrame(self.time_sync_tab, text="Connection Details", padding=(10, 5))
        input_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(input_frame, text="Host:").grid(row=0, column=0, sticky="w", pady=2)
        self.ts_host_entry = ttk.Entry(input_frame, width=30)
        self.ts_host_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.ts_host_entry.insert(0, "rs.local")
        ttk.Label(input_frame, text="Username:").grid(row=1, column=0, sticky="w", pady=2)
        self.ts_username_entry = ttk.Entry(input_frame, width=30)
        self.ts_username_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.ts_username_entry.insert(0, "myshake")
        ttk.Label(input_frame, text="Password:").grid(row=2, column=0, sticky="w", pady=2)
        self.ts_password_entry = ttk.Entry(input_frame, show="*", width=30)
        self.ts_password_entry.grid(row=2, column=1, sticky="ew", padx=5)
        input_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(self.time_sync_tab)
        button_frame.pack(pady=5)

        self.connect_button = ttk.Button(button_frame, text="Connect", command=self.run_connect)
        self.connect_button.pack(side="left", padx=5)

        self.sync_time_button = ttk.Button(button_frame, text="Sync Time", command=self.run_sync_time, state="disabled")
        self.sync_time_button.pack(side="left", padx=5)

        self.disconnect_button = ttk.Button(button_frame, text="Disconnect", command=self.run_disconnect, state="disabled")
        self.disconnect_button.pack(side="left", padx=5)

        # Status Indicator
        self.ts_status_label = ttk.Label(self.time_sync_tab, text="Status: Idle", anchor="center")
        self.ts_status_label.pack(fill="x", padx=10, pady=5)
        output_frame = ttk.LabelFrame(self.time_sync_tab, text="Output", padding=(10, 5))
        output_frame.pack(padx=10, pady=(0, 10), expand=True, fill="both")
        self.ts_output_text = scrolledtext.ScrolledText(output_frame, width=70, height=10, wrap=tk.WORD)
        self.ts_output_text.pack(expand=True, fill="both")

    def run_connect(self):
        logging.info("'Connect' button clicked.")
        host = self.ts_host_entry.get()
        username = self.ts_username_entry.get()
        password = self.ts_password_entry.get()
        if not all([host, username, password]):
            logging.error("Connection input error: all fields are required.")
            messagebox.showerror("Input Error", "All fields are required.")
            return
        
        self.shake_communicator = ShakeCommunicator(host, username, password)
        
        logging.info(f"Attempting to connect to {host} for user {username}.")
        self.connect_button.config(state="disabled")
        self.update_ts_status("Connecting...", "blue")
        self.ts_output_text.delete('1.0', tk.END)
        self.ts_output_text.insert(tk.INSERT, f"Attempting to connect to {host}...\n\n")
        self.start_task(self.connect_worker)

    def connect_worker(self):
        try:
            result = self.shake_communicator.connect()
            logging.info(f"Connection result for {self.shake_communicator.host}: {result}")
            self.task_queue.put((self.on_connect_result, result))
        except Exception as e:
            logging.error(f"Connection error for {self.shake_communicator.host}: {e}", exc_info=True)
            self.task_queue.put((self.handle_error, "Connection Error", e))

    def on_connect_result(self, result):
        self.ts_output_text.insert(tk.END, result + "\n")
        if "successful" in result.lower():
            self.update_ts_status("Connected", "green")
            self.sync_time_button.config(state="normal")
            self.disconnect_button.config(state="normal")
            self.connect_button.config(state="disabled")
        else:
            self.update_ts_status("Failed", "red")
            self.connect_button.config(state="normal")
            self.shake_communicator = None

    def run_disconnect(self):
        logging.info("'Disconnect' button clicked.")
        self.disconnect_button.config(state="disabled")
        self.sync_time_button.config(state="disabled")
        self.update_ts_status("Disconnecting...", "blue")
        self.start_task(self.disconnect_worker)

    def disconnect_worker(self):
        try:
            result = self.shake_communicator.disconnect()
            logging.info(f"Disconnection result: {result}")
            self.task_queue.put((self.on_disconnect_result, result))
        except Exception as e:
            logging.error(f"Disconnection error: {e}", exc_info=True)
            self.task_queue.put((self.handle_error, "Disconnect Error", e))

    def on_disconnect_result(self, result):
        self.ts_output_text.insert(tk.END, result + "\n")
        self.update_ts_status("Disconnected", "red")
        self.connect_button.config(state="normal")
        self.sync_time_button.config(state="disabled")
        self.disconnect_button.config(state="disabled")
        self.shake_communicator = None

    def run_sync_time(self):
        logging.info("'Sync Time' button clicked.")
        self.sync_time_button.config(state="disabled")
        self.update_ts_status("Syncing time...", "blue")
        self.ts_output_text.insert(tk.INSERT, "Attempting to sync time...\n\n")
        self.start_task(self.sync_time_worker)

    def sync_time_worker(self):
        try:
            result = self.shake_communicator.set_time_utc()
            logging.info(f"Time sync result: {result}")
            self.task_queue.put((self.on_sync_time_result, result))
        except Exception as e:
            logging.error(f"Time sync error: {e}", exc_info=True)
            self.task_queue.put((self.handle_error, "Time Sync Error", e))

    def on_sync_time_result(self, result):
        self.ts_output_text.insert(tk.END, result + "\n")
        self.sync_time_button.config(state="normal")
        if "Error" in result:
            self.update_ts_status("Error", "red")
        else:
            self.update_ts_status("Connected", "green")

    def update_ts_status(self, status, color):
        self.ts_status_label.config(text=f"Status: {status}", foreground=color)

    # --- Single Fetch Tab ---
    def create_data_acquisition_tab(self):
        input_frame = ttk.LabelFrame(self.data_acquisition_tab, text="Waveform Parameters", padding=(10, 5))
        input_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(input_frame, text="Host:").grid(row=0, column=0, sticky="w", pady=2)
        self.da_host_entry = ttk.Entry(input_frame, width=30)
        self.da_host_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.da_host_entry.insert(0, "rs.local")
        ttk.Label(input_frame, text="Port:").grid(row=1, column=0, sticky="w", pady=2)
        self.da_port_entry = ttk.Entry(input_frame, width=30)
        self.da_port_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.da_port_entry.insert(0, "16032")
        ttk.Label(input_frame, text="Network:").grid(row=2, column=0, sticky="w", pady=2)
        self.da_net_entry = ttk.Entry(input_frame, width=30)
        self.da_net_entry.grid(row=2, column=1, sticky="ew", padx=5)
        self.da_net_entry.insert(0, "AM")
        ttk.Label(input_frame, text="Station:").grid(row=3, column=0, sticky="w", pady=2)
        self.da_sta_entry = ttk.Entry(input_frame, width=30)
        self.da_sta_entry.grid(row=3, column=1, sticky="ew", padx=5)
        self.da_sta_entry.insert(0, "R1E3F")
        ttk.Label(input_frame, text="Location:").grid(row=4, column=0, sticky="w", pady=2)
        self.da_loc_entry = ttk.Entry(input_frame, width=30)
        self.da_loc_entry.grid(row=4, column=1, sticky="ew", padx=5)
        self.da_loc_entry.insert(0, "00")
        ttk.Label(input_frame, text="Channel:").grid(row=5, column=0, sticky="w", pady=2)
        self.da_cha_entry = ttk.Entry(input_frame, width=30)
        self.da_cha_entry.grid(row=5, column=1, sticky="ew", padx=5)
        self.da_cha_entry.insert(0, "EH*")
        
        # Start Time
        ttk.Label(input_frame, text="Start Time (UTC):").grid(row=6, column=0, sticky="w", pady=2)
        start_time_frame = ttk.Frame(input_frame)
        start_time_frame.grid(row=6, column=1, sticky="ew", padx=5)
        self.da_start_entry = ttk.Entry(start_time_frame, width=25)
        self.da_start_entry.pack(side="left", fill="x", expand=True)
        start_btn = ttk.Button(start_time_frame, text="...", width=3, command=lambda: self.open_datetime_picker(self.da_start_entry))
        start_btn.pack(side="left")

        # End Time
        ttk.Label(input_frame, text="End Time (UTC):").grid(row=7, column=0, sticky="w", pady=2)
        end_time_frame = ttk.Frame(input_frame)
        end_time_frame.grid(row=7, column=1, sticky="ew", padx=5)
        self.da_end_entry = ttk.Entry(end_time_frame, width=25)
        self.da_end_entry.pack(side="left", fill="x", expand=True)
        end_btn = ttk.Button(end_time_frame, text="...", width=3, command=lambda: self.open_datetime_picker(self.da_end_entry))
        end_btn.pack(side="left")

        # Set default times
        now = datetime.now(timezone.utc)
        start_time = now.strftime("%Y-%m-%dT%H:%M:%S")
        end_time = (now + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
        self.da_start_entry.insert(0, start_time)
        self.da_end_entry.insert(0, end_time)

        input_frame.columnconfigure(1, weight=1)
        button_frame = ttk.Frame(self.data_acquisition_tab)
        button_frame.pack(pady=5)
        self.get_waveforms_button = ttk.Button(button_frame, text="Get Waveforms", command=self.run_get_waveforms)
        self.get_waveforms_button.pack(side="left", padx=5)
        self.plot_waveforms_button = ttk.Button(button_frame, text="Plot Waveforms", command=self.plot_waveforms)
        self.plot_waveforms_button.pack(side="left", padx=5)
        output_frame = ttk.LabelFrame(self.data_acquisition_tab, text="Output", padding=(10, 5))
        output_frame.pack(padx=10, pady=(0, 10), expand=True, fill="both")
        self.da_output_text = scrolledtext.ScrolledText(output_frame, width=70, height=10, wrap=tk.WORD)
        self.da_output_text.pack(expand=True, fill="both")
        self.stream = None

    def open_datetime_picker(self, entry_widget):
        DateTimePicker(self.root, entry_widget)

    def run_get_waveforms(self):
        try:
            params = {
                "host": self.da_host_entry.get(), "port": int(self.da_port_entry.get()),
                "net": self.da_net_entry.get(), "sta": self.da_sta_entry.get(),
                "loc": self.da_loc_entry.get(), "cha": self.da_cha_entry.get(),
                "start_time": UTCDateTime(self.da_start_entry.get()),
                "end_time": UTCDateTime(self.da_end_entry.get())
            }
            self.get_waveforms_button.config(state="disabled")
            self.da_output_text.delete('1.0', tk.END)
            self.da_output_text.insert(tk.INSERT, f"Connecting to {params['host']}:{params['port']}...\n")
            self.start_task(self.get_waveforms_worker, params)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            self.get_waveforms_button.config(state="normal")

    def get_waveforms_worker(self, params):
        try:
            self.task_queue.put((self.update_da_output, f"Fetching waveforms for {params['net']}.{params['sta']}.{params['loc']}.{params['cha']}...\n"))
            stream = fetch_waveforms(params)
            self.task_queue.put((self.finish_get_waveforms, stream))
        except Exception as e:
            self.task_queue.put((self.handle_error, "Waveform Fetch Error", e))

    def finish_get_waveforms(self, stream):
        self.stream = stream
        self.da_output_text.insert(tk.INSERT, "Waveforms fetched successfully.\n")
        self.da_output_text.insert(tk.INSERT, str(self.stream) + "\n")
        self.get_waveforms_button.config(state="normal")
        output_file = filedialog.asksaveasfilename(defaultextension=".mseed", filetypes=[("MSEED files", "*.mseed")])
        if output_file:
            try:
                self.stream.write(output_file, format="MSEED",)
                self.da_output_text.insert(tk.INSERT, f"Stream saved to {output_file}\n")
                logging.info(f"Stream successfully saved to {output_file}")
            except Exception as e:
                logging.error(f"Failed to save stream to {output_file}: {e}", exc_info=True)
                messagebox.showerror("File Save Error", f"Failed to save file: {e}")

    def update_da_output(self, text):
        self.da_output_text.insert(tk.INSERT, text)

    def plot_waveforms(self):
        if self.stream:
            self.stream.plot()
        else:
            messagebox.showinfo("No Data", "No waveform data to plot. Please fetch waveforms first.")

    def handle_error(self, title, error):
        messagebox.showerror(title, str(error))
        self.run_mhvsr_button.config(state="normal")
        if title == "Connection Error":
            self.connect_button.config(state="normal")
            self.update_ts_status("Failed", "red")
        elif title == "Disconnect Error":
            self.connect_button.config(state="normal")
            self.update_ts_status("Disconnected", "red")
        elif title == "Time Sync Error":
            self.sync_time_button.config(state="normal")
            self.update_ts_status("Error", "red")
        elif title == "Waveform Fetch Error":
            self.get_waveforms_button.config(state="normal")

    # --- Multifetch Tab ---
    def create_multifetch_tab(self):
        # Main frame for the tab
        main_frame = ttk.Frame(self.multifetch_tab)
        main_frame.pack(fill="both", expand=True)

        # --- Top Setup Frame ---
        setup_frame = ttk.Frame(main_frame)
        setup_frame.pack(fill="x", padx=10, pady=5)

        # Project Setup
        project_frame = ttk.LabelFrame(setup_frame, text="Project Setup", padding=(10, 5))
        project_frame.pack(fill="x", side="left", expand=True, padx=(0, 5))

        ttk.Label(project_frame, text="Project Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.mf_project_name_entry = ttk.Entry(project_frame)
        self.mf_project_name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5)
        self.mf_project_name_entry.insert(0, f"Project_{datetime.now(timezone.utc).strftime('%Y%m%d')}")

        ttk.Label(project_frame, text="Project Directory:").grid(row=1, column=0, sticky="w", pady=2)
        self.mf_project_dir_entry = ttk.Entry(project_frame)
        self.mf_project_dir_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.mf_project_dir_button = ttk.Button(project_frame, text="Browse...", command=self.select_project_directory)
        self.mf_project_dir_button.grid(row=1, column=2, sticky="w", padx=5)
        
        ttk.Label(project_frame, text="Number of Stations:").grid(row=2, column=0, sticky="w", pady=2)
        self.mf_station_count_spinbox = ttk.Spinbox(project_frame, from_=1, to=1000, width=7)
        self.mf_station_count_spinbox.grid(row=2, column=1, sticky="w", padx=5)
        
        self.mf_set_stations_button = ttk.Button(project_frame, text="Generate Station Inputs", command=self.generate_station_inputs)
        self.mf_set_stations_button.grid(row=2, column=2, padx=5, pady=5)
        
        project_frame.columnconfigure(1, weight=1)

        # --- Connection Details Frame ---
        conn_frame = ttk.LabelFrame(setup_frame, text="Shake Connection Details", padding=(10, 5))
        conn_frame.pack(fill="x", side="right", expand=True, padx=(5, 0))

        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, sticky="w", pady=2)
        self.mf_host_entry = ttk.Entry(conn_frame)
        self.mf_host_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.mf_host_entry.insert(0, "rs.local")

        ttk.Label(conn_frame, text="Port:").grid(row=1, column=0, sticky="w", pady=2)
        self.mf_port_entry = ttk.Entry(conn_frame)
        self.mf_port_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.mf_port_entry.insert(0, "16032")

        ttk.Label(conn_frame, text="Network:").grid(row=2, column=0, sticky="w", pady=2)
        self.mf_net_entry = ttk.Entry(conn_frame)
        self.mf_net_entry.grid(row=2, column=1, sticky="ew", padx=5)
        self.mf_net_entry.insert(0, "AM")

        ttk.Label(conn_frame, text="Station:").grid(row=3, column=0, sticky="w", pady=2)
        self.mf_sta_entry = ttk.Entry(conn_frame)
        self.mf_sta_entry.grid(row=3, column=1, sticky="ew", padx=5)
        self.mf_sta_entry.insert(0, "R1E3F")

        ttk.Label(conn_frame, text="Location:").grid(row=4, column=0, sticky="w", pady=2)
        self.mf_loc_entry = ttk.Entry(conn_frame)
        self.mf_loc_entry.grid(row=4, column=1, sticky="ew", padx=5)
        self.mf_loc_entry.insert(0, "00")

        ttk.Label(conn_frame, text="Channel:").grid(row=5, column=0, sticky="w", pady=2)
        self.mf_cha_entry = ttk.Entry(conn_frame)
        self.mf_cha_entry.grid(row=5, column=1, sticky="ew", padx=5)
        self.mf_cha_entry.insert(0, "EH*")
        
        conn_frame.columnconfigure(1, weight=1)

        # --- Frame to hold the scrollable station inputs ---
        canvas_frame = ttk.LabelFrame(main_frame, text="Station Time Windows", padding=(10, 5))
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.stations_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.stations_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.stations_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.stations_canvas.configure(scrollregion=self.stations_canvas.bbox("all"))
        )

        self.stations_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.stations_canvas.configure(yscrollcommand=scrollbar.set)

        self.stations_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.station_widgets = []

        # --- Bottom frame for controls and output ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

        self.mf_fetch_all_button = ttk.Button(bottom_frame, text="Fetch All Waveforms", command=self.run_multifetch)
        self.mf_fetch_all_button.pack(pady=5)

        output_frame = ttk.LabelFrame(bottom_frame, text="Output", padding=(10, 5))
        output_frame.pack(fill="both", expand=True)
        self.mf_output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD)
        self.mf_output_text.pack(expand=True, fill="both")

    def select_project_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.mf_project_dir_entry.delete(0, tk.END)
            self.mf_project_dir_entry.insert(0, directory)

    def generate_station_inputs(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.station_widgets = []

        try:
            num_stations = int(self.mf_station_count_spinbox.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Input Error", "Number of stations must be a valid integer.")
            return

        for i in range(num_stations):
            station_frame = ttk.Frame(self.scrollable_frame, padding=5)
            station_frame.pack(fill="x", padx=5, pady=5, expand=True)
            
            ttk.Label(station_frame, text=f"Station {i+1}:").grid(row=0, column=0, sticky="w")
            
            # Start Time
            ttk.Label(station_frame, text="Start Time (UTC):").grid(row=0, column=1, sticky="w", padx=(10, 0))
            start_time_entry = ttk.Entry(station_frame, width=22)
            start_time_entry.grid(row=0, column=2, sticky="ew")
            start_btn = ttk.Button(station_frame, text="...", width=3, command=lambda e=start_time_entry: self.open_datetime_picker(e))
            start_btn.grid(row=0, column=3, padx=5)

            # End Time
            ttk.Label(station_frame, text="End Time (UTC):").grid(row=0, column=4, sticky="w", padx=(10, 0))
            end_time_entry = ttk.Entry(station_frame, width=22)
            end_time_entry.grid(row=0, column=5, sticky="ew")
            end_btn = ttk.Button(station_frame, text="...", width=3, command=lambda e=end_time_entry: self.open_datetime_picker(e))
            end_btn.grid(row=0, column=6, padx=5)
            
            now = datetime.now(timezone.utc)
            start_time_entry.insert(0, now.strftime("%Y-%m-%dT%H:%M:%S"))
            end_time_entry.insert(0, (now + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S"))

            self.station_widgets.append({"start": start_time_entry, "end": end_time_entry})
            station_frame.columnconfigure(2, weight=1)
            station_frame.columnconfigure(5, weight=1)

    def run_multifetch(self):
        project_name = self.mf_project_name_entry.get()
        project_dir = self.mf_project_dir_entry.get()

        if not project_name or not project_dir:
            messagebox.showerror("Input Error", "Project Name and Project Directory are required.")
            return

        if not self.station_widgets:
            messagebox.showerror("Input Error", "Please generate station inputs first.")
            return
        
        try:
            base_params = {
                "host": self.mf_host_entry.get(), "port": int(self.mf_port_entry.get()),
                "net": self.mf_net_entry.get(), "sta": self.mf_sta_entry.get(),
                "loc": self.mf_loc_entry.get(), "cha": self.mf_cha_entry.get(),
            }
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input Error", f"Invalid Shake Connection Details: {e}")
            return

        all_params = []
        for i, station in enumerate(self.station_widgets):
            try:
                start_time = UTCDateTime(station["start"].get())
                end_time = UTCDateTime(station["end"].get())
                
                # Create a copy of base_params and update it
                params = base_params.copy()
                params.update({
                    "start_time": start_time,
                    "end_time": end_time,
                    "station_num": i + 1
                })
                all_params.append(params)
            except Exception as e:
                messagebox.showerror("Input Error", f"Invalid date/time for Station {i+1}: {e}")
                return
                
        self.mf_fetch_all_button.config(state="disabled")
        self.mf_output_text.delete('1.0', tk.END)
        self.mf_output_text.insert(tk.INSERT, f"Starting multifetch for project: {project_name}\n")
        logging.info(f"Starting multifetch for project: {project_name}")
        
        self.start_task(self.multifetch_worker, project_name, project_dir, all_params)

    def multifetch_worker(self, project_name, project_dir, all_params):
        try:
            if not os.path.exists(project_dir):
                self.task_queue.put((self.update_mf_output, f"Project directory not found. Please select a valid directory.\n"))
                return
            
            project_path = os.path.join(project_dir, project_name)
            os.makedirs(project_path, exist_ok=True)
        except Exception as e:
            self.task_queue.put((self.handle_error, "Directory Error", f"Could not create project directory: {e}"))
            return

        for params in all_params:
            station_num = params["station_num"]
            self.task_queue.put((self.update_mf_output, f"\n--- Fetching Station {station_num} ---\n"))
            try:
                stream = fetch_waveforms(params)
                self.task_queue.put((self.update_mf_output, f"  Successfully fetched {len(stream)} traces.\n"))
                
                # Create filename: projectname_stationnumber_starttime_endtime.mseed
                st_str = params['start_time'].strftime('%Y%m%dT%H%M%S')
                et_str = params['end_time'].strftime('%Y%m%dT%H%M%S')
                filename = f"{project_name}_{station_num}_{st_str}_to_{et_str}.mseed"
                output_file = os.path.join(project_path, filename)
                
                stream.write(output_file, format="MSEED")
                self.task_queue.put((self.update_mf_output, f"  Saved stream to {filename}\n"))
                logging.info(f"Saved stream for station {station_num} to {output_file}")
                
            except Exception as e:
                error_msg = f"  Error for station {station_num}: {e}\n"
                self.task_queue.put((self.update_mf_output, error_msg))
                logging.error(f"Error fetching/saving station {station_num}: {e}", exc_info=True)

        self.task_queue.put((self.finish_multifetch, "\n--- Multifetch complete! ---\n"))

    def update_mf_output(self, text):
        self.mf_output_text.insert(tk.INSERT, text)
        self.mf_output_text.see(tk.END)

    def finish_multifetch(self, text):
        self.update_mf_output(text)
        self.mf_fetch_all_button.config(state="normal")

    # --- MHVSR Analysis Tab ---
    def create_mhvsr_tab(self):
        main_frame = ttk.Frame(self.mhvsr_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Top Frame ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=5)

        # --- File Selection ---
        file_frame = ttk.LabelFrame(top_frame, text="Input Files", padding=(10, 5))
        file_frame.pack(fill="x", expand=True, side="left", padx=(0, 5))

        self.mhvsr_files = []
        self.mhvsr_file_list_var = tk.StringVar(value="No files selected.")
        ttk.Label(file_frame, text="Selected Files:").pack(side="left", padx=5)
        ttk.Label(file_frame, textvariable=self.mhvsr_file_list_var, wraplength=300).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(file_frame, text="Select Files", command=self.select_mhvsr_files).pack(side="right", padx=5)

        # --- Parameters ---
        param_frame = ttk.LabelFrame(top_frame, text="HVSR Parameters", padding=(10, 5))
        param_frame.pack(fill="x", expand=True, side="right", padx=(5, 0))

        ttk.Label(param_frame, text="Window Length (s):").grid(row=0, column=0, sticky="w", pady=2)
        self.mhvsr_window_length = tk.StringVar(value="150")
        ttk.Spinbox(param_frame, from_=1, to=1000, width=7, textvariable=self.mhvsr_window_length).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(param_frame, text="K&O Bandwidth:").grid(row=0, column=2, sticky="w", pady=2, padx=(10,0))
        self.mhvsr_bandwidth = tk.StringVar(value="40")
        ttk.Spinbox(param_frame, from_=1, to=100, width=7, textvariable=self.mhvsr_bandwidth).grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(param_frame, text="Filter Low Cut (Hz):").grid(row=1, column=0, sticky="w", pady=2)
        self.mhvsr_filter_low = tk.StringVar(value="None")
        ttk.Entry(param_frame, width=9, textvariable=self.mhvsr_filter_low).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(param_frame, text="Filter High Cut (Hz):").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.mhvsr_filter_high = tk.StringVar(value="None")
        ttk.Entry(param_frame, width=9, textvariable=self.mhvsr_filter_high).grid(row=1, column=3, sticky="w", padx=5)

        ttk.Label(param_frame, text="Taper Type:").grid(row=2, column=0, sticky="w", pady=2)
        self.mhvsr_taper_type = tk.StringVar(value="tukey")
        taper_options = ["tukey", "hann", "hamming", "bartlett", "blackman"]
        ttk.Combobox(param_frame, textvariable=self.mhvsr_taper_type, values=taper_options, state="readonly", width=7).grid(row=2, column=1, sticky="ew", padx=5)

        ttk.Label(param_frame, text="Taper Width (alpha):").grid(row=2, column=2, sticky="w", pady=2, padx=(10,0))
        self.mhvsr_taper_width = tk.StringVar(value="0.2")
        ttk.Spinbox(param_frame, from_=0, to=1, increment=0.05, width=7, textvariable=self.mhvsr_taper_width).grid(row=2, column=3, sticky="w", padx=5)

        ttk.Label(param_frame, text="Combine Horizontals:").grid(row=3, column=0, sticky="w", pady=2)
        self.mhvsr_combine_method = tk.StringVar(value="geometric_mean")
        combine_options = ["geometric_mean", "squared_average", "azimuth", "single_azimuth"]
        ttk.Combobox(param_frame, textvariable=self.mhvsr_combine_method, values=combine_options, state="readonly").grid(row=3, column=1, columnspan=3, sticky="ew", padx=5)

        # --- Analysis and Output ---
        analysis_frame = ttk.Frame(main_frame)
        analysis_frame.pack(fill="both", expand=True, pady=5)

        # --- Controls ---
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill="x")

        self.run_mhvsr_button = ttk.Button(control_frame, text="Run MHVSR Analysis", command=self.run_mhvsr_analysis)
        self.run_mhvsr_button.pack(side="left", padx=5)

        self.plot_mhvsr_button = ttk.Button(control_frame, text="Plot Results", command=self.plot_mhvsr_results, state="disabled")
        self.plot_mhvsr_button.pack(side="left", padx=5)

        self.save_mhvsr_button = ttk.Button(control_frame, text="Save Results", command=self.save_mhvsr_results, state="disabled")
        self.save_mhvsr_button.pack(side="left", padx=5)

        # --- Output ---
        output_frame = ttk.LabelFrame(analysis_frame, text="Output", padding=(10, 5))
        output_frame.pack(fill="both", expand=True, pady=5)

        self.mhvsr_output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD)
        self.mhvsr_output_text.pack(expand=True, fill="both")

        self.hvsr_result = None

    def select_mhvsr_files(self):
        files = filedialog.askopenfilenames(title="Select MSEED/MiniSEED Files", filetypes=[("MSEED/MiniSEED files", "*.mseed *.miniseed"), ("All files", "*.*")])
        if files:
            self.mhvsr_files = files
            self.mhvsr_file_list_var.set(f"{len(files)} files selected.")

    def run_mhvsr_analysis(self):
        if not self.mhvsr_files:
            messagebox.showerror("Input Error", "Please select input files first.")
            return

        self.run_mhvsr_button.config(state="disabled")
        self.plot_mhvsr_button.config(state="disabled")
        self.save_mhvsr_button.config(state="disabled")
        self.mhvsr_output_text.delete('1.0', tk.END)
        self.mhvsr_output_text.insert(tk.INSERT, "Running MHVSR analysis...\n")

        self.start_task(self.mhvsr_worker)

    def mhvsr_worker(self):
        try:
            window_length = int(self.mhvsr_window_length.get())
            bandwidth = int(self.mhvsr_bandwidth.get())
            combine_method = self.mhvsr_combine_method.get()
            
            # New parameters
            low_cut_str = self.mhvsr_filter_low.get()
            high_cut_str = self.mhvsr_filter_high.get()
            taper_type = self.mhvsr_taper_type.get()
            taper_width = float(self.mhvsr_taper_width.get())

            low_cut = float(low_cut_str) if low_cut_str.lower() != 'none' else None
            high_cut = float(high_cut_str) if high_cut_str.lower() != 'none' else None

            preprocessing_settings = get_default_preprocessing_settings()
            preprocessing_settings.window_length_in_seconds = window_length
            preprocessing_settings.filter_corner_frequencies_in_hz = (low_cut, high_cut)

            processing_settings = get_default_processing_settings()
            processing_settings.window_type_and_width = (taper_type, taper_width)
            processing_settings.smoothing['bandwidth'] = bandwidth
            processing_settings.method_to_combine_horizontals = combine_method

            hvsr = process_mhvsr([list(self.mhvsr_files)], preprocessing_settings, processing_settings)
            self.task_queue.put((self.on_mhvsr_complete, hvsr))
        except Exception as e:
            self.task_queue.put((self.handle_error, "MHVSR Error", e))

    def on_mhvsr_complete(self, hvsr):
        self.hvsr_result = hvsr
        self.mhvsr_output_text.insert(tk.INSERT, "MHVSR analysis complete.\n")
        self.run_mhvsr_button.config(state="normal")
        self.plot_mhvsr_button.config(state="normal")
        self.save_mhvsr_button.config(state="normal")

        # Display summary
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            hvsr.update_peaks_bounded(search_range_in_hz=(None, None))
            print("\nSESAME (2004) Clarity and Reliability Criteria:")
            print("-"*47)
            hvsrpy.sesame.reliability(
                windowlength=int(self.mhvsr_window_length.get()),
                passing_window_count=np.sum(hvsr.valid_window_boolean_mask),
                frequency=hvsr.frequency,
                mean_curve=hvsr.mean_curve(distribution="lognormal"),
                std_curve=hvsr.std_curve(distribution="lognormal"),
                search_range_in_hz=(None, None),
                verbose=1,
            )
            hvsrpy.sesame.clarity(
                frequency=hvsr.frequency,
                mean_curve=hvsr.mean_curve(distribution="lognormal"),
                std_curve=hvsr.std_curve(distribution="lognormal"),
                fn_std=hvsr.std_fn_frequency(distribution="normal"),
                search_range_in_hz=(None, None),
                verbose=1,
            )
            print("\nStatistical Summary:")
            print("-"*20)
            hvsrpy.summarize_hvsr_statistics(hvsr)
        s = f.getvalue()
        self.mhvsr_output_text.insert(tk.INSERT, s)


    def plot_mhvsr_results(self):
        if self.hvsr_result:
            import matplotlib.pyplot as plt
            fig, ax = hvsrpy.plot_single_panel_hvsr_curves(self.hvsr_result)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            fig.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
        else:
            messagebox.showinfo("No Data", "No MHVSR results to plot. Please run the analysis first.")

    def save_mhvsr_results(self):
        if self.hvsr_result:
            output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_file:
                try:
                    hvsrpy.object_io.write_hvsr_object_to_file(self.hvsr_result, output_file)
                    self.mhvsr_output_text.insert(tk.INSERT, f"\nResults saved to {output_file}\n")
                    logging.info(f"MHVSR results successfully saved to {output_file}")
                except Exception as e:
                    logging.error(f"Failed to save MHVSR results to {output_file}: {e}", exc_info=True)
                    messagebox.showerror("File Save Error", f"Failed to save file: {e}")
        else:
            messagebox.showinfo("No Data", "No MHVSR results to save. Please run the analysis first.")



if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = ShakeFetchApp(root)
    root.mainloop()