from textual.app import ComposeResult, App, on
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, RichLog
from textual.binding import Binding
import asyncio

class CompanionApp(App):
    """Main TUI app"""
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #main-container {
        height: 100%;
        border: solid $accent;
    }
    
    #chat-display {
        height: 1fr;
        border-bottom: solid $accent;
    }
    
    #input-area {
        height: auto;
        padding: 1;
    }
    
    Input {
        width: 100%;
    }
    """
    
    def __init__(self, chat_loop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_loop = chat_loop
        self.chat_log = None
        self.input = None
    
    def compose(self) -> ComposeResult:
        yield Vertical(
            RichLog(id="chat-display", markup=True),
            Horizontal(
                Input(id="user-input", placeholder="Type your message..."),
                id="input-area"
            ),
            id="main-container"
        )
    
    def on_mount(self):
        """Called when app starts"""
        self.chat_log = self.query_one("#chat-display", RichLog)
        self.input = self.query_one("#user-input", Input)
        self.input.focus()
        self.chat_log.write("[bold green]Alex:[/bold green] Hey! What's on your mind today?")
    
    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input"""
        user_msg = event.value
        self.input.value = ""
        
        # Display user message
        self.chat_log.write(f"[bold cyan]You:[/bold cyan] {user_msg}")
        
        # Get response from chat loop
        try:
            response = await self.chat_loop.get_response(user_msg)
            self.chat_log.write(f"[bold green]Alex:[/bold green] {response}")
        except Exception as e:
            self.chat_log.write(f"[bold red]Error:[/bold red] {str(e)}")