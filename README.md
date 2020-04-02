# Janken Hockey Game

[![License](https://img.shields.io/github/license/skmatz/janken-hockey-game)](LICENSE)

**Janken + Hockey**  
(In Japanese, rock-paper-scissors is called janken.)

## Overview

Janken Hockey Game is a hockey game that can play between 2-PCs.

The feature of this game is that players play hockey while playing rock-paper-scissors.

## How to Play

### Preparation

```sh
git clone https://github.com/skmatz/janken-hockey-game.git
cd janken-hockey-game
poetry install
wget https://skmatz-weights.s3-ap-northeast-1.amazonaws.com/jhgame.weights -O models/jhgame.weights
```

### Server PC (Only 1 PC)

```sh
poetry run python run.py --server
```

### Client PC (All PCs to play)

```sh
poetry run python run.py
```

## Requirement

- `Python >= 3.6`

## Note

This project is an updated version of the game created by students at Taniguchi Laboratory in Tokyo University of Science for display on the 2019 Open Campus.

## Credit

Thank you for these awesome projects.

- [pygame](https://github.com/pygame/pygame)
- [pygame-menu](https://github.com/ppizarror/pygame-menu/)
