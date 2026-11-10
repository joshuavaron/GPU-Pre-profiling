CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -std=c11 -fPIC -pthread
BUILD = build

all: $(BUILD)/libmemprof.a

$(BUILD):
	mkdir -p $@

$(BUILD)/memprof.o: memprof.c memprof.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ memprof.c

$(BUILD)/libmemprof.a: $(BUILD)/memprof.o
	ar rcs $@ $^

$(BUILD)/test_memprof: tests/test_memprof.c $(BUILD)/libmemprof.a
	$(CC) $(CFLAGS) -I. -o $@ $< $(BUILD)/libmemprof.a

test: $(BUILD)/test_memprof
	./$(BUILD)/test_memprof $(BUILD)/example_report.json
	@if command -v python3 >/dev/null; then \
	  python3 -m json.tool $(BUILD)/example_report.json >/dev/null && echo "example_report.json: valid JSON"; \
	fi

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
