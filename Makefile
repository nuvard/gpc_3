build/filter: main.cc linear-algebra.hh reduce-scan.hh filter.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp main.cc -lOpenCL -o build/filter
