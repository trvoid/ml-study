from pickle import dump, load

t1 = (1, 2)

with open('x.pickle', 'wb') as f:
    dump(t1, f)

print(f'Write Done: {t1}')

with open('x.pickle', 'rb') as f:
    t2 = load(f)

print(f'Read Done: {t2}')
