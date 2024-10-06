from tkinter import *
from tkinter import ttk
import customtkinter
from customtkinter import filedialog
from tkinter import messagebox
import EarthquakeDetection

class SeismoMiraclesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SeismoMiracles")
        self.root.geometry("800x240")

        self.name = "Please select mseed"
        
        # Configure appearance
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Label for file name
        self.fileNameLabel = customtkinter.CTkLabel(master=self.root, text=self.name, font=customtkinter.CTkFont(size=20, weight="bold"))
        self.fileNameLabel.place(relx=0.5, rely=0.2, anchor=customtkinter.CENTER)
        
        # Button for selecting file
        self.selectFileButton = customtkinter.CTkButton(master=self.root, text="Select file", command=self.select_file)
        self.selectFileButton.place(relx=0.5, rely=0.4, anchor=customtkinter.CENTER)
        
        # Option menu for appearance mode
        self.appearanceModeOptionMenu = customtkinter.CTkOptionMenu(master=self.root, values=["Moon", "Mars"])
        self.appearanceModeOptionMenu.place(relx=0.5, rely=0.6, anchor=customtkinter.CENTER)
        
        # Button for analysis
        self.analyzeButton = customtkinter.CTkButton(master=self.root, text="Analyze", command=self.analyze)
        self.analyzeButton.place(relx=0.5, rely=0.8, anchor=customtkinter.CENTER)

    def select_file(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.name = filename
            self.fileNameLabel.configure(text=self.name)

    def analyze(self):
        if self.name.endswith('.mseed'):
            selected_value = self.appearanceModeOptionMenu.get()
            if 'mars' in selected_value.lower():
                mars_analyzer = EarthquakeDetection.Mars(self.name)
                mars_analyzer.main()
            elif 'moon' in selected_value.lower():
                moon_analyzer = EarthquakeDetection.Moon(self.name)
                moon_analyzer.main()
        else:
            messagebox.showerror("Invalid File", "Please select a valid .mseed file.")

if __name__ == "__main__":
    app = customtkinter.CTk()
    SeismoMiraclesApp(app)
    app.mainloop()
