# --- Escapes ------------------------------------------------------------------
# handles system and terminal specific commands for dealing with blocking,
# echoing, and canonical modes, returning focus, and modifying color, cursor
# and keys for terminal input and output


# --- Imports ------------------------------------------------------------------

# Standard libraries
import sys, tty, os, termios, fcntl


# --- Blocking -----------------------------------------------------------------

class BlockingEchoCanonicalOff():
    """blocking, echoing, and canonical modes are turned off"""
    def __enter__(self):
        self.attributes = termios.tcgetattr(sys.stdin) # original attributes

        # modify original attributes to non canonical mode, with no echo
        newAttributes = self.attributes
        newAttributes[3] = newAttributes[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, newAttributes)

        self.flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL) # save original flags

        # modify original flags to non blocking mode
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self.flags | os.O_NONBLOCK)

    def __exit__(self, *args):
        # restore terminal to previous state
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.attributes)
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self.flags)

class BlockingOn():
    """blocking mode is turned on"""
    def __enter__(self):
        self.flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL) # save original flags

        # modify original flags to blocking mode
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self.flags | ~os.O_NONBLOCK)

    def __exit__(self, *args):
        # restore terminal to previous state
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self.flags)


# --- Terminal Handling --------------------------------------------------------

def returnFocus(terminal="iTerm2"):
    """control returns to specified terminal application running the script"""
    path = "/usr/bin/osascript" # path to osascript
    os.system(f"""{path} -e 'tell application "{terminal}" to activate'""")

def hideCursor():
    """cursor becomes invisible"""
    print("\033[?25l", end='') # print hide cursor escape string

def showCursor():
    """cursor becomes visible"""
    print("\033[?25h", end='') # print show cursor escape string

def clearLine():
    """clear all characters from current line"""
    print("\033[0K", end='') # print clear line escape string

def reset():
    """color and style are reset to original values"""
    print("\033[0m", end='') # print reset color escape string

def bold():
    """change print text to bold style"""
    print("\033[1m", end='')

def moveCursor(n, direction):
    """move the cursor n steps in direction specified"""

    switcher = { # dictionary mapping direction to escape string
        "up": "\033[" + str(n) + "A",
        "down": "\033[" + str(n) + "B",
        "right": "\033[" + str(n) + "C",
        "left": "\033[" + str(n) + "D"
    }
    print(switcher.get(direction, "error"), end='') # print escape string

def changeColor(r, g, b, ground='f'):
    """change print color to RGB values for the foreground or background"""

    if ground == "background" or ground == 'b': # change background color
        print(f"\033[48;2;{r};{g};{b}m", end='')
    elif ground == "foreground" or ground == 'f': # change foreground color
        print(f"\033[38;2;{r};{g};{b}m", end='')
    else: # otherwise error
        print("Error: unrecognized ground input")

def shutdown():
    """reset everything before quitting"""
    os.system('stty sane')
    showCursor()
    reset()
    print("Quit sanely." + " "*20)

def getKey():
    """reads the key input from stdin and returns the string name of the key"""
    keyMap = { # dictionary mapping key code ints to key name strings
        127: "backspace",
        10:  "return",
        32:  "space",
        9:   "tab",
        27:  "esc",
        65:  "up",
        66:  "down",
        67:  "right",
        68:  "left"
    }

    k = sys.stdin.read(3)
    if len(k) == 0:
        return ''
    if len(k) == 2:
        k = ord(k[1])
    elif len(k) == 3:
        k = ord(k[2])
    else:
        k = ord(k)
    return keyMap.get(k, chr(k))
