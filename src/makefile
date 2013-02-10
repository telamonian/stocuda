CC=g++ -pg
CFLAGS=  -std=c++11 -g  -O0 -c -Wall
PROF =

all: gil

gil: hazard.o pnet.o main.o
	$(CC) main.o pnet.o hazard.o -pg -o gil $(PROF)

hazard.o: hazard.cc hazard.hh
	$(CC) $(CFLAGS) hazard.cc $(PROF)

pnet.o: pnet.cc pnet.hh hazard.hh
	$(CC) $(CFLAGS) pnet.cc hazard.o $(PROF)

main.o: main.cc
	$(CC) $(CFLAGS) main.cc pnet.o hazard.o $(PROF)

clean:
	rm -rf *o gi