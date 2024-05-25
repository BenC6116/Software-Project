CC=gcc
CFLAGS=-ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS=-lm
SRC=symnmf.c
EXECUTABLE=symnmf

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(SRC) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)