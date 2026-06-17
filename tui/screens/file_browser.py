"""Interactive file/directory browser screen.

Keyboard-navigable directory tree with Enter to confirm selection.
Supports filtering by file extension and directory-only mode.
"""
import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label


class FileBrowserScreen(Screen):
    """Interactive file browser with keyboard navigation.

    - Arrow keys / j,k: navigate list
    - Enter: select item (descend into dir or confirm file)
    - Backspace: go to parent directory
    - ESC: cancel and return
    - Ctrl+H: go to home directory
    - Type path prefix to filter (handled via ListView search)

    Returns the selected path via dismiss(result).
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("backspace", "go_parent", "Parent Dir"),
        ("ctrl+h", "go_home", "Home Dir"),
    ]

    CSS = """
    FileBrowserScreen {
        align: center middle;
    }
    #browser-container {
        width: 85%;
        height: 90%;
        border: solid $primary;
        padding: 1 2;
    }
    #browser-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    #browser-path {
        color: $warning;
        padding: 0 0 1 0;
        text-style: italic;
    }
    #browser-hint {
        color: $text-muted;
        padding-bottom: 1;
    }
    #browser-list {
        height: 1fr;
        border: solid $surface;
    }
    #browser-status {
        color: $text-muted;
        padding: 1 0;
        min-height: 1;
    }
    Button {
        margin: 0 1;
    }
    """

    def __init__(self, start_path: str = ".", ext_filter: str | None = None,
                 dirs_only: bool = False, name: str | None = None,
                 id: str | None = None, classes: str | None = None):
        super().__init__(name=name, id=id, classes=classes)
        self._start_path = Path(start_path).resolve()
        self._ext_filter = ext_filter  # e.g. ".nc", ".csv", ".pt", ".pth"
        self._dirs_only = dirs_only
        self._current_dir = self._start_path
        if not self._current_dir.exists():
            self._current_dir = Path.home()
        if self._current_dir.is_file():
            self._current_dir = self._current_dir.parent

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="browser-container"):
            yield Static("File Browser", id="browser-title")
            yield Static(str(self._current_dir), id="browser-path")
            yield Static(
                "Enter=Select  Backspace=Parent  ESC=Cancel  Ctrl+H=Home",
                id="browser-hint",
            )
            yield ListView(id="browser-list")
            yield Static("", id="browser-status")
            with Horizontal():
                yield Label("Quick jump:")
                yield Static("Press / to search filter", id="browser-search-hint")
        yield Footer()

    def on_mount(self):
        self._refresh_list()

    def _refresh_list(self):
        """Refresh the directory listing."""
        list_view = self.query_one("#browser-list", ListView)
        list_view.clear()

        self.query_one("#browser-path", Static).update(str(self._current_dir))

        try:
            entries = sorted(self._current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            self.query_one("#browser-status", Static).update("[Error] Permission denied")
            return

        filtered = []
        for p in entries:
            if p.name.startswith("."):
                continue
            if p.is_dir():
                filtered.append((p, True))
            elif not self._dirs_only:
                if self._ext_filter is None or p.suffix in self._ext_filter:
                    filtered.append((p, False))

        if not filtered:
            list_view.append(ListItem(Label("(empty directory)")))
            self.query_one("#browser-status", Static).update("No matching items in current directory")
            return

        for path, is_dir in filtered:
            prefix = "📁 " if is_dir else "📄 "
            name = path.name
            if not is_dir:
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    suffix = f"  ({size_mb:.1f} MB)" if size_mb >= 0.1 else ""
                except OSError:
                    suffix = ""
            else:
                try:
                    count = sum(1 for _ in path.iterdir())
                    suffix = f"  ({count} 项)"
                except (PermissionError, OSError):
                    suffix = ""
            item = ListItem(Label(f"{prefix}{name}{suffix}"))
            item.metadata = {"path": path, "is_dir": is_dir}
            list_view.append(item)

        self.query_one("#browser-status", Static).update(
            f"{len(filtered)} items  |  Current: {self._current_dir}"
        )

    def _get_selected(self) -> dict | None:
        """Get the currently selected item's metadata."""
        list_view = self.query_one("#browser-list", ListView)
        if list_view.index is None:
            return None
        items = list_view.children
        valid_items = [w for w in items if isinstance(w, ListItem) and hasattr(w, "metadata") and w.metadata]
        if list_view.index < len(valid_items):
            return valid_items[list_view.index].metadata
        return None

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle Enter on a list item."""
        if event.item is None:
            return
        meta = getattr(event.item, "metadata", None)
        if meta is None:
            return

        if meta.get("is_dir"):
            self._current_dir = meta["path"]
            self._refresh_list()
        else:
            self.dismiss(str(meta["path"]))

    def dismiss(self, result=None):
        """Store result on app before dismissing."""
        if result is not None:
            self.app._browser_result = result
        else:
            self.app._browser_result = None
        super().dismiss()

    def action_cancel(self):
        """Cancel and return to previous screen."""
        self.app._browser_result = None
        super().dismiss()

    def action_go_parent(self):
        """Navigate to parent directory."""
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._refresh_list()

    def action_go_home(self):
        """Navigate to home directory."""
        self._current_dir = Path.home()
        self._refresh_list()
