import click

from data2vec_vision.train import train


@click.group("Data2Vec vision")
def _d2v() -> None:
    pass


if __name__ == "__main__":
    _d2v.add_command(train, "train")
    _d2v()
