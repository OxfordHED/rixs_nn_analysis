import click
from .figure_5 import figure_5
from .figure_6 import figure_6

@click.group()
def app():
    """
    Plot the figures seen in the main text of the paper.
    """
    pass


app.add_command(figure_5)
app.add_command(figure_6)

if __name__ == "__main__":
    app()
