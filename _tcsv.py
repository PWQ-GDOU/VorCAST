import csv
fn="2/20230102.csv"
with open(fn) as f:
    r=csv.reader(f)
    hdr=next(r)
    h=[x.strip() for x in hdr]
    si=h.index("Storm Number")
    ti=h.index("Time")
    next(r)  # skip units
    storm1000=[]
    for row in r:
        try:
            sn=int(row[si].strip())
        except:
            continue
        if sn==1000:
            storm1000.append(row[ti].strip())
print(f"File: {fn}")
print(f"Storm 1000 entries: {len(storm1000)}")
if storm1000:
    print(f"Times: {storm1000[:5]}...{storm1000[-3:]}")
# Also check for any 20230104 dates
count_04=0
with open(fn) as f:
    r=csv.reader(f)
    next(r); next(r)
    for row in r:
        if '20230104' in row[ti]:
            count_04+=1
print(f"Total 20230104 entries in this file: {count_04}")
