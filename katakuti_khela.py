# This is the KataKuti game Version:0 to be played against a machine by Arghya Chattopadhyay



# With helps from "realpython.com/tic-tac-toe-python/" for the basic maize and the game logic.
# Future version will include
# --- A time prompt to get a time dependent game
# --- An improved version of the AI

# The following libraries are for the katakuti maze and the game play
import tkinter as tk
from tkinter import font
from itertools import cycle
from typing import NamedTuple # The data class for the player and the moves

# Calling horsemen of tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Libraries to do some real calculations
import numpy as np

# To track the flow of time for future, but for now to manage different files
import time, os, sys

# To save the game history in csv format
import csv
import pandas

############################## Character Development ##############################

class Khelowar(NamedTuple):
    label: str # To hold the X or O sign
    color: str # To identify the target player on the board
    name: str
class Daan(NamedTuple):
    row: int # row coordinate of the target move
    col: int # column coordinate of the target move
    label: str="" # Empty string denoting that this move is not played yet

############################## Basic variables ##############################

BOARD_SIZE= 3 # can be scaled to any integer value
Manus=Khelowar(label="X", color="blue", name="Human")
Machine=Khelowar(label="O", color="green", name="Machine")
DEFAULT_KHELOWARS=(Manus,Machine)
DICTO={'':0,'X':1,'O':2} # This asigned values are needed for the network to learn


############################## Global Functions ##############################


# To generate the winning combos

def get_winning_combos():
    rows=[[(row,col) for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
    columns=[list(col) for col in zip(*rows)]
    first_diagonal=[row[i] for i, row in enumerate(rows)]
    second_diagonal=[col[j] for j, col in enumerate(reversed(columns))]
    return rows + columns + [first_diagonal, second_diagonal]

WINNING_COMBOS=get_winning_combos()

# Another short function for the neural network utilisation for fixing the correct output

def vectorized_result(j):
    e = [[0]*(BOARD_SIZE**2)]
    e[0][j] = 1.0
    return e


############################## Game file retention ##############################

# To write/rewrite the history of the game
def historian(file_name, total_games=0, human_wins=0, machine_wins=0, draws=0, read=False):
    if read:
        hist = pandas.read_csv(file_name)
        return hist["Total games Played"][0], hist["Human Wins"][0],hist["Machine Wins"][0], hist["Draws"][0]
        
    lekho=open(file_name, 'w', newline='')
    writer=csv.writer(lekho)
    writer.writerow(["Total games Played","Human Wins","Machine Wins","Draws"])
    writer.writerow([total_games, human_wins, machine_wins,draws])
    lekho.close()
    
def to_row_col(number):
    return divmod(number,BOARD_SIZE)

def to_number(row,col):
    return BOARD_SIZE*row+col


############################## Agent with a Network ##############################


class katakuti_net(keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.hid1=keras.layers.Dense(hidden_size//2, activation=tf.nn.tanh)
        self.hid2=keras.layers.Dense(hidden_size, activation=tf.nn.tanh)
        self.hid3=keras.layers.Dense(hidden_size//2, activation=tf.nn.tanh)
        self.hid=keras.layers.Dense(BOARD_SIZE**2, activation=tf.nn.softmax)
        
    def call(self,input_data):
        x=self.hid1(input_data)
        x=self.hid2(x)
        x=self.hid3(x)
        hid=self.hid(x)
        return hid

class Agent():
    def __init__(self):
        self.mod_opt=tf.keras.optimizers.Adam(learning_rate=1.0)
        self._error_lim=1e-5
        self._hidden_size=32
        self.encode_net=katakuti_net(self._hidden_size)

        
    def gen_chal(self, current_state):
        self.state=tf.Variable(initial_value=current_state,trainable=None,dtype=float)
        self._out=self.encode_net(self.state)
        return self._out
    
    
    def gen_chal_fine(self, current_state,n):
        # to make the n-th highest probable state as the correct answer
        self.state=tf.Variable(initial_value=current_state,trainable=None,dtype=float)
        with tf.GradientTape(persistent=True) as grand_tape:
            self._out=self.encode_net(self.state)
            next_out=tf.Variable(initial_value=vectorized_result(np.argsort(self._out[0])[-n]),trainable=None,dtype=float)
            self._cost=tf.square(next_out-self._out)
        grads=grand_tape.gradient(self._cost,self.encode_net.trainable_variables)
        self.mod_opt.apply_gradients(zip(grads,self.encode_net.trainable_variables))
        
    

############################## The Game Logic ##############################


# The following class will take care of the moves, toggle between players and declare winners as well
class katakutikhela:
    def __init__(self, khelowars=DEFAULT_KHELOWARS, board_size=BOARD_SIZE):
        self._khelowars=cycle(khelowars) # A cyclical iterator
        self.board_size=board_size
        self.current_khelowar=next(self._khelowars)
        self.winner_combo=[]
        self.ai=Agent()
        self._current_chal=[]
        self._current_vec=[]
        self._has_winner=False # This variable is always checked
        self._winning_combos=[] # Cell combinations that defines a win
        # One should note that this whole game can be generalised to any size and therefore the winning combos
        # have to be calculated at the first instance of the game
        self._setup_board()
        self.if_exist_load_ai()
    
    def if_exist_load_ai(self):
        if os.path.exists('AI/katakuti_net_BoardSize_{0}'.format(BOARD_SIZE)):
            self.ai.encode_net=tf.keras.models.load_model('AI/katakuti_net_BoardSize_{0}'.format(BOARD_SIZE))
            print("Existing AI found and loaded")
        
        
    def _setup_board(self):
        self._current_chal=[
            [Daan(row,col) for col in range(self.board_size)] for row in range(self.board_size)]
        self._current_vec=[[0]*(BOARD_SIZE**2)]
        self._winning_combos=WINNING_COMBOS
        
    def convert_to_vec(self):
        for row in self._current_chal:
            for daan in row:
                self._current_vec[0][to_number(daan.row,daan.col)]=DICTO[daan.label]
        
    def is_valid_chal(self,daan):
        # gets the players input as daan
        row, col= daan.row, daan.col
        daan_was_not_played= self._current_chal[row][col].label==""
        no_winner=not self._has_winner
        return no_winner and daan_was_not_played
    
    def let_ai_play(self, valid=False):
        self._ai_probs=np.argsort(self.ai.gen_chal(self._current_vec)[0])
        print(self._ai_probs)
        order=1
        while True:
            ai_row,ai_col=to_row_col(self._ai_probs[-order])
            print(order,")The AI is choosing:",ai_row,ai_col)
            valid=self.is_valid_chal(Daan(ai_row,ai_col,Machine.label))
            if valid:
                print("The AI is playing with:",ai_row,ai_col)
                self.ai.gen_chal_fine(self._current_vec,order)
                break
            print("The AI chose the wrong values")
            order=order+1
        
        return ai_row,ai_col
        
    
    def process_chal(self, daan):
        # Process the current move and check for winners
        row, col= daan.row, daan.col
        self._current_chal[row][col]= daan
        for combo in self._winning_combos:
            results=set(
                self._current_chal[n][m].label
                for n,m in combo
            )
            # set object cannot hold duplicate items
            is_win=(len(results)==1) and ("" not in results)
            if is_win:
                self._has_winner=True
                self.winner_combo=combo
                break
    
    def has_winner(self):
        return self._has_winner
    
    def is_tied(self):
        no_winner=not self._has_winner
        played_chal=( daan.label for row in self._current_chal for daan in row )
        return no_winner and all(played_chal)
    
    def toggle_khelowar(self):
        self.current_khelowar=next(self._khelowars)
        
    def reset_khela(self):
        for row, row_content in enumerate(self._current_chal):
            for col,_ in enumerate(row_content):
                row_content[col]=Daan(row,col)
        self._has_winner=False
        self.winner_combo=[]
        self.convert_to_vec()


############################## The Game Board ##############################

# This class designs the main game looks and feels


class katakutiboard(tk.Tk):
    def __init__(self,game):
        super().__init__()
        self.title("Kata-Kuti-Khela")
        self._cells={}
        self._clicks={}
        self._game=game
        self._total_game, self._human_win, self._machine_win, self._draws= historian(game_data,read=True)
        self._create_menu()
        self._create_board_chehara()
        self._create_board_chhok()
    
                
    def _create_board_chehara(self):
        chehara_frame=tk.Frame(master=self) # Creates the frame object to hold the display
        chehara_frame.pack(fill=tk.X) 
        # geometry manager: fill ensures that frame fills the screen under resize
        # Below creates the label object to live inside the frame object
        self.chehara=tk.Label(
            master=chehara_frame,
            text=f"{self._game.current_khelowar.name}'s turn", # The initial display
            font=font.Font(size=28, weight="bold"), #Play with the font family in future version
        )
        self.chehara.pack() # Label object is now inside the chehara   
   

    def _create_board_chhok(self):
        chhok_frame=tk.Frame(master=self) # To hold the frame for the games grid
        chhok_frame.pack()
        for row in range(self._game.board_size):
            # Following configures the width and minimum size of every cell on the grid
            self.rowconfigure(row, weight=1, minsize=50)
            self.columnconfigure(row, weight=1, minsize=50)
            for col in range(self._game.board_size):
                button=tk.Button(
                    master=chhok_frame,
                    text="",
                    font=font.Font(size=36, weight="bold"),
                    width=3,
                    height=2,
                    highlightbackground="lightblue",
                )
                self._cells[button]=(row,col)
                button.bind("<ButtonPress-1>", self.play) # Clicking a button will run the game
                # Above adds every new button to the ._cells dictionary. Buttons works as keys and 
                # their coordinates or values expressed as (row,col).
                button.grid(
                    row=row,
                    column=col,
                    padx=5,
                    pady=5,
                    sticky="nsew"
                )
        self._clicks= {v: k for k, v in self._cells.items()}
    def _create_menu(self):
        menu_bar=tk.Menu(master=self)
        self.config(menu=menu_bar)
        file_menu= tk.Menu(master=menu_bar)
        file_menu.add_command(
            label="New Match",
            command= self.reset_board
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit",command=quit)
        menu_bar.add_cascade(label="Options", menu=file_menu)
        
    def _update_button(self, clicked_btn):
        clicked_btn.config(text=self._game.current_khelowar.label)
        clicked_btn.config(fg=self._game.current_khelowar.color)
        
        
    def _update_chehara(self, msg, color="black"):
        self.chehara["text"]=msg
        self.chehara["fg"]=color
        
    def _highlight_cells(self):
        for button, coordinates in self._cells.items():
            if coordinates in self._game.winner_combo:
                button.config(highlightbackground="red")
                
    def play(self,event):
        
        if self._game.current_khelowar.name=='Human':
            clicked_btn=event.widget # Tkinter event object
            row, col= self._cells[clicked_btn]
            print("The human chose:", row, col)
            self.chal_processing(row,col,clicked_btn)
            print("The human turn is processed")
            
        if self._game.current_khelowar.name=='Machine':
            row, col=self._game.let_ai_play()
            print("The AI chose:", row, col)
            clicked_btn=self._clicks[(row,col)]
            self.chal_processing(row,col,clicked_btn)
            print("The AI turn is processed")
            
    def chal_processing(self,row,col,clicked_btn): 
        daan=Daan(row, col, self._game.current_khelowar.label)
        if self._game.is_valid_chal(daan):
            self._update_button(clicked_btn)
            self._game.process_chal(daan)
            if self._game.is_tied():
                self._total_game+=1
                self._draws+=1
                self._update_chehara(msg=f"Match Drawn!\n Scoreboard \n Machine: {self._machine_win} \n Human: {self._human_win}\n Drawn: {self._draws}", color="red")
                self._game.ai.encode_net.save('AI/katakuti_net_BoardSize_{0}'.format(BOARD_SIZE))
                
                historian(game_data,self._total_game, self._human_win, self._machine_win, self._draws)
            elif self._game.has_winner():
                self._highlight_cells()
                self._total_game+=1
                self._human_win+=1
                if self._game.current_khelowar.name=='Machine':
                    self._machine_win+=1
                    self._human_win-=1
                msg= f"{self._game.current_khelowar.name} won the match \n Scoreboard \n Machine: {self._machine_win} \n Human: {self._human_win}\n Drawn: {self._draws}"
                color= self._game.current_khelowar.color
                self._update_chehara(msg, color)

                self._game.ai.encode_net.save('AI/katakuti_net_BoardSize_{0}'.format(BOARD_SIZE))
                historian(game_data,self._total_game, self._human_win, self._machine_win, self._draws)
            else:
                self._game.toggle_khelowar()
                msg= f"{self._game.current_khelowar.name}'s turn"
                self._update_chehara(msg)
                
    def reset_board(self):
        self._game.reset_khela()
        self._update_chehara(msg=f"{self._game.current_khelowar.name}'s turn")
        for button in self._cells.keys():
            button.config(highlightbackground="lightblue")
            button.config(text="")
            button.config(fg="black")
    

############################## A file folder handler ##############################


# Getting the file system prepared for the AI to save itslef

main_folder=".Game_Data" # Remove the dot in the begining to make this folder visible
game_data="history.csv"

if not os.path.exists(main_folder):
    os.mkdir(main_folder)

# Now let us go inside the main_folder to check further
os.chdir(main_folder)

# Following file holds the game history
if not os.path.isfile(game_data):
    historian(game_data)

if not os.path.exists("AI"):
    os.mkdir("AI")


############################## The main infinite loop ##############################



def main(): # Creates the main loop of the game
    game=katakutikhela()
    board=katakutiboard(game)
    board.mainloop()

# To allow to call the main() only when executed but not imported
if __name__=="__main__":
    main()




############################## End of file ##############################