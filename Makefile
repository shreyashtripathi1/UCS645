CC = mpicc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
TARGETS = q1 q2 q3 q4 q5

.PHONY: all run clean report

all: $(TARGETS)

q1: q1.c
	$(CC) $(CFLAGS) -o q1 q1.c $(LDFLAGS)

q2: q2.c
	$(CC) $(CFLAGS) -o q2 q2.c

q3: q3.c
	$(CC) $(CFLAGS) -o q3 q3.c $(LDFLAGS)

q4: q4.c
	$(CC) $(CFLAGS) -o q4 q4.c $(LDFLAGS)

q5: q5.c
	$(CC) $(CFLAGS) -o q5 q5.c $(LDFLAGS)

# Run all programs
run: all
	@echo "=========================================="
	@echo "Running Question 1: DAXPY Operation"
	@echo "=========================================="
	mpirun -np 2 ./q1
	@echo
	@echo "=========================================="
	@echo "Running Question 2: Broadcast Race"
	@echo "=========================================="
	mpirun -np 4 ./q2
	@echo
	@echo "=========================================="
	@echo "Running Question 3: Distributed Dot Product"
	@echo "=========================================="
	mpirun -np 2 ./q3
	@echo
	@echo "=========================================="
	@echo "Running Question 4: Prime Number Finder"
	@echo "=========================================="
	mpirun -np 4 ./q4
	@echo
	@echo "=========================================="
	@echo "Running Question 5: Perfect Number Finder"
	@echo "=========================================="
	mpirun -np 4 ./q5

# Compile only
compile: $(TARGETS)
	@echo "All programs compiled successfully"

# Clean
clean:
	rm -f $(TARGETS) *.o report.txt
	@echo "Cleaned up all executables"

help:
	@echo "Available targets:"
	@echo "  all      - Compile all programs"
	@echo "  run      - Compile and run all programs"
	@echo "  compile  - Compile only (same as 'all')"
	@echo "  clean    - Remove all compiled files"
	@echo "  help     - Show this help message"
