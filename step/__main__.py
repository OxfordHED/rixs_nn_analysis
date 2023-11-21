import click
from .train import train
from .eval import eval

@click.group()
def app():
    pass


app.add_command(train)
app.add_command(eval)

if __name__ == "__main__":
    app()
