import click
from .train import train
from .test import test

@click.group()
def app():
    pass


app.add_command(train)
app.add_command(test)

if __name__ == "__main__":
    app()
