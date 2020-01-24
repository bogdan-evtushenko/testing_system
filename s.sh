# compile all programs first
g++ brute.cpp -o brute
g++ a.cpp -o a
g++ gen.cpp -o gen

for((i = 1; ; ++i)); do
    echo $i
    {
    	./gen $i > int
    	./a < int > out1
    } || {
        break
    }
    ./brute < int > out2
    diff -w out1 out2 || break
    diff -w <(./a < int) <(./brute < int) || break
done
