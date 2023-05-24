import click
from .train import train
from .test import test

@click.group()
def app():
    """CLI to interact with end-to-end models.
    """
    pass


app.add_command(train)
app.add_command(test)

if __name__ == "__main__":
    app()
