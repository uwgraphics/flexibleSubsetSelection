# --- Imports ------------------------------------------------------------------

# Standard library
import os

# Local files
from . import escapes

# --- Screens ------------------------------------------------------------------


class Screen:
    def __init__(self, title):
        """create a blank screen with a title"""
        self.title = title
        self.blankScreen()

    def blankScreen(self):
        """create a new blank text screen with a title"""
        outer = 4
        innerSpace = 3

        self.lineToWrite = 2
        self.buttons = []
        self.selected = 0
        self.width, self.height = os.get_terminal_size()  # get height and width

        innerWidth = self.width - 2
        remainingWidth = self.width - 4 - outer - innerSpace * 2 - len(self.title)

        # line 1
        self.content = f"{' ' * (outer + 1)}"  # title box left indent
        self.content += f"╔{'═' * (len(self.title) + innerSpace * 2)}╗"  # box top
        self.content += f"{' ' * (remainingWidth + 1)}\n"  # title box right indent

        # line 2
        self.content += f"┌{'─' * outer}"  # title box left indent
        self.content += f"║{' ' * innerSpace}{self.title}{' ' * innerSpace}║"
        self.content += f"{'─' * remainingWidth}┐\n"  # title box right indent

        # line 3
        self.content += f"│{' ' * outer}"  # title box left indent
        self.content += f"╚{'═' * (len(self.title) + innerSpace * 2)}╝"  # box base
        self.content += f"{' ' * remainingWidth}│\n"  # title box right indent

        # remaining lines
        self.content += f"│{' ' * innerWidth}│\n" * (self.height - 4)  # box sides
        self.content += f"└{'─' * innerWidth}┘"  # screen box base

    def writeLine(self, text, x=10, y=None):
        """write text on screen at column x line y"""
        if y is None:
            y = self.lineToWrite
        self.lineToWrite += 1

        start = (3 + y) * (self.width + 1) + x + 1
        end = start + len(text)
        self.content = f"{self.content[:start]}{text}{self.content[end:]}"

    def writeLog(self, text=""):
        """write specified text to log space or clear by not specifying text"""
        text = " " * (50 - len(text)) + text
        x = self.width - 56
        y = self.height - 6
        self.writeLine(text, x, y)  # write log message
        self.print()  # print updated screen

    def addButton(self, button):
        """save a Button to the Screen and write it's text to the Screen"""
        self.buttons.append(button)
        if not button.y:  # if button does not have y position,
            button.y = self.lineToWrite  # place it on next available line
        button.write(self)

    def addBackButton(self, function):
        start = (self.height - 5) * (self.width + 1)
        end = start + self.width
        divider = "─" * (self.width - 2)
        self.content = f"{self.content[:start]}├{divider}┤{self.content[end:]}"
        self.addButton(Button("ᐊ Back", function, x=4, y=self.height - 6))

    def print(self):
        """print the Screen to the terminal display"""
        self.clear()
        pos = 0
        for index, button in enumerate(self.buttons):
            start = (3 + button.y) * (self.width + 1) + button.x
            end = start + len(button.text) + 2
            print(self.content[pos:start], end="")
            if index == self.selected:
                escapes.changeColor(141, 211, 199, "background")
                escapes.changeColor(0, 0, 0, "foreground")
            print(self.content[start:end], end="")
            if button.__class__.__name__ == "TypedInput":
                escapes.bold()
                start = end
                end += len(str(button.value)) + 2
                print(self.content[start:end], end="")
            escapes.reset()
            pos = end
        print(self.content[end:], end="")

    def clear(self):
        """clear the terminal display"""
        escapes.moveCursor(self.height, "up")
        escapes.moveCursor(self.width, "left")

    def keyDown(self):
        """move the selection cursor down when down key is pressed"""
        if self.selected < len(self.buttons) - 1:
            self.selected += 1
            self.clear()
            self.print()

    def keyUp(self):
        """move the selection cursor up when up key is pressed"""
        if self.selected > 0:
            self.selected -= 1
            self.clear()
            self.print()

    def keyReturn(self):
        """if there is a button selected, run its function"""
        if len(self.buttons) > 0:
            button = self.buttons[self.selected]
            return button.activate(self)


# --- Menus --------------------------------------------------------------------


class Menu:
    def __init__(self, title, buttons):
        self.title = title
        self.buttons = buttons

    def write(self, screen):
        outerMargin = 8
        innerMargin = 2
        width = screen.width - 2 * outerMargin - 4
        remainingWidth = width - len(self.title) - innerMargin - 4

        start = (2 + screen.lineToWrite) * (screen.width + 1)
        content = screen.content[:start]

        content += f"│{' ' * (outerMargin + innerMargin + 1)}"
        content += f"┌{'─' * (len(self.title) + 2)}┐"
        content += f"{' ' * (remainingWidth + outerMargin + 1)}│\n"

        content += f"│{' ' * outerMargin}"
        content += f"┌{'─' * innerMargin}│ {self.title} │{'─' * remainingWidth}┐"
        content += f"{' ' * outerMargin}│\n"

        content += f"│{' ' * outerMargin}│"
        content += f"{' ' * innerMargin}└{'─' * (len(self.title) + 2)}┘"
        content += f"{' ' * remainingWidth}│{' ' * outerMargin}│\n"

        for i in range(len(self.buttons)):
            content += f"│{' ' * outerMargin}│{' ' * width}│{' ' * outerMargin}│\n"

        content += f"│{' ' * outerMargin}└{'─' * (width)}┘{' ' * outerMargin}│\n"
        end = (6 + len(self.buttons) + screen.lineToWrite) * (screen.width + 1)
        content += screen.content[end:]
        screen.content = content
        screen.lineToWrite += 2
        for button in self.buttons:
            screen.addButton(button)
        screen.lineToWrite += 3


# --- Buttons ------------------------------------------------------------------


class Button:
    def __init__(self, text, function, x=14, y=None):
        self.text = text
        self.function = function
        self.x = x
        self.y = y

    def write(self, screen):
        screen.writeLine(self.text, self.x, self.y)

    def activate(self, *_):
        self.function()
        return None, None


class TypedInput(Button):
    def __init__(self, text, parameter, value):
        if value.__class__.__name__ in ["int", "int64"]:
            function = str.isnumeric
        elif value.__class__.__name__ == "str":
            function = str.isalpha
        elif value.__class__.__name__ == "range":
            function = str.isalpha

        Button.__init__(self, text, function)
        self.value = str(value)
        self.parameter = parameter

    def write(self, screen):
        screen.writeLine(f"{self.text}: {self.value}", self.x, self.y)

    def activate(self, screen):
        """handle typed input"""
        value = self.value
        self.value = " " * len(value)
        self.write(screen)
        screen.clear()
        screen.print()

        self.value = ""
        self.write(screen)
        screen.clear()
        screen.print()

        with escapes.BlockingEchoCanonicalOff():
            while True:
                k = escapes.getKey()
                with escapes.BlockingOn():
                    if k == "return":
                        break
                    elif k == "backspace":
                        self.value = self.value[:-1]
                        self.value += " "
                        self.write(screen)
                        screen.print()
                        self.value = self.value[:-1]
                    elif self.function(k):
                        self.value += k
                    self.write(screen)
                    screen.print()

        if self.value == "":
            self.value = value
            self.write(screen)
            screen.clear()
            screen.print()

        if self.function == str.isalpha:
            self.value = f"'{self.value}'"
        return self.parameter, self.value


class BooleanButton(Button):
    def __init__(self, text, function, parameter, value):
        Button.__init__(self, text, function)
        self.parameter = parameter
        self.value = value
        if self.value:
            self.text = f"◉ {self.text}"
        else:
            self.text = f"○ {self.text}"

    def update(self, screen):
        if self.value:
            self.text = f"◉{self.text[1:]}"
        else:
            self.text = f"○{self.text[1:]}"
        self.write(screen)

    def activate(self, screen):
        self.value = not self.value
        self.update(screen)
        screen.print()
        self.function(self.parameter, self.value)
        return None, None


class RadioButton(BooleanButton):
    def __init__(self, text, function, parameter, option, value):
        Button.__init__(self, text, function)
        self.parameter = parameter
        self.option = option
        if value == option:
            self.value = True
            self.text = f"◉ {self.text}"
        else:
            self.value = False
            self.text = f"○ {self.text}"

    def activate(self, screen):
        self.value = True
        self.update(screen)
        self.function(self.parameter, self.option)
        return None, None
