import shutil

def insert_data (latitude,longtitude,type):
    with open("my_data.csv", mode="at") as f:
        f.write(','.join(map(str,[latitude,longtitude,type])) +'\n')
        f.close()

def get_data (latitude,longtitude):
    with open("my_data.csv", mode="r") as f:
        for line in f:
            data = line.strip().split(',')
            if str(data[0]) == str(latitude) and str(data[1]) == str(longtitude):
                return str(data[2])
        f.close()
        return 'none'

def update_data (latitude,longtitude,type):
    with open("my_data.csv", mode="r") as f1:
        with open("my_data_bak.csv", mode="wt") as f2:
            for line in f1:
                data = line.strip().split(',')
                if str(data[0]) == str(latitude) and str(data[1]) == str(longtitude):
                    f2.write(','.join(map(str,[latitude,longtitude,type])) +'\n')
                else:
                    f2.write(line)
            f2.close()
        f1.close()
        shutil.move("my_data_bak.csv", "my_data.csv")


#insert_data (32.1,21.7,'land')
#insert_data (33.2,22.9,'sea')

test = get_data(32.1,21.7)
print(test)

update_data(32.1,21.7,'pond')
