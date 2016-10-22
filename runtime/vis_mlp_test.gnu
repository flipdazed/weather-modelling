### Expected arguments

### files
base_loc = "dump/data/".run_id."_"
ext = ".dat"

cost_files = system("ls ".base_loc."cost_"."*".ext)
params_files = system("ls ".base_loc."params_"."*".ext)

params_names = system("ls ".base_loc."params_"."*".ext." | sed -e 's#".base_loc."params_"."\\(.*\\)\\".ext."#\\1#' -e 's#_# #g'")

#function used to map a value to the intervals
hist(x,width)=width*floor(x/width)+width/2.0

### Start multiplot
set terminal wxt size 1440,800 font 'Verdana,6'

set multiplot
xn=4
yn = 1+1+2
x = xn
y = yn
o = 0.1

xi = o
yi = o

xf = x-o
xs = xf-xi
yf = y-o
ys = yf-yi

# --- GRAPH cost
unset key
set title "cost vs. training samples" font ",10"
f = word(cost_files, 1)
set size xs/x,ys/(yn/2)/y
set origin xi/x,(yi+ys/(yn/2))/y
set xlabel "sample" font ",7"
set ylabel "value" font ",7"
plot f u (column(0)):1 w l ls 1 lc rgb"blue"

set xtics rotate by -45
# --- GRAPH params per batch
do for [i=1:words(params_files)] {
    unset key
    p = word(params_names,i*2-1)." ".word(params_names,i*2)
    f = word(params_files, i)
    set title p." vs. training samples" font ",10"
    set size (xs/xn)/x,(ys/yn)/y
    set origin (xi+(i-1)*(xs/xn))/x,(yi+ys/yn)/y
    set xlabel "sample" font ",7"
    set ylabel "mean value" font ",7"
    plot f u (column(0)):1 with lines ls 1
}

# --- GRAPH histogram of params
do for [i=1:words(params_files)] {
    unset key
    p = word(params_names,i*2-1)." ".word(params_names,i*2)
    f = word(params_files, i)

    set title "freq. of av. ".p font ",10"

    set size (xs/xn)/x,(ys/yn)/y
    set origin (xi+(i-1)*(xs/xn))/x,yi/y
    stats f using 1 nooutput

    n=50 #number of intervals
    max=STATS_mean+3*STATS_stddev #max value
    min=STATS_mean-3*STATS_stddev #min value
    width=(max-min)/n #interval width
    set xlabel "mean value" font ",7"
    set ylabel "freq" font ",7"
    plot f u (hist($1,width)):(1.0/STATS_records) \
    smooth freq w l lc rgb"green" notitle
}

unset multiplot
### End multiplot

pause 2
reread