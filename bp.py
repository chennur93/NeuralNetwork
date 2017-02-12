import random
import math
import copy

def sigmoid(a):    
    return 1/(1+math.exp(-1*(a)))

def modify(y):
    r=max(y)
    for i in range(len(y)):
        if y[i]<r:
            y[i]=0
        else:
            y[i]=1
    return y

def pos(y):
    for i in range(len(y)):
        if y[i]==max(y):
            return i

def main():
    #read input
    file=open("t1.csv","r")
    r=file.readline()
    s1=64#no. of units in layer 1
    s2=64#no. of units in hidden layer
    inp=[]
    out=[]
    input_size=0
    while r!='':
        temp=r.split(',')
        inp.append([int(temp[i]) for i in range(0, len(temp)-1)])
        k=[0 for i in range(len(temp))]
        k[int(temp[len(temp)-1])]=1
        out.append(k)
        r=file.readline()
        input_size+=1

    count=0
    
    #weights 65x64 at layer 1
    weights_l1=[[random.random() for i in range(len(inp[0])+1)] for j in range(len(inp[0]))]
    #weights 65x64 at layer 2
    weights_l2=[[random.random() for i in range(len(inp[0])+1)] for j in range(len(inp[0]))]
    #weights 65x10 at layer 3
    weights_l3=[[random.random() for i in range(len(inp[0])+1)] for j in range(10)]
    #do for every row in the input
    for row in range(input_size):
        
        #compute values at the hidden layer
        o=[]
        for i in range(s2):#for each unit in hidden layer
            r=weights_l1[i][0]#bias
            for j in range(s2):
               r+= weights_l1[i][j+1]*float(inp[row][j])
            r=sigmoid(r)
            o.append(r)

        o1=[]
        for i in range(s2):
            r=weights_l2[i][0]
            for j in range(s2):
                r+=weights_l2[i][+1]*o[j]
            r=sigmoid(r)
            o1.append(r)


        #now, we have values at the hidden layer
        #we then compute the output values
        y=[]
        for i in range(10):
            r=weights_l3[i][0]
            for j in range(s2):
               r+= weights_l3[i][j+1]*o1[j]
            r=sigmoid(r)
            y.append(r)
        y=[float(i)/sum(y) for i in y]

        if pos(y)==pos(out[row]):
            count+=1
            print(pos(y))
        #adjust weights l3
        del_weights_l3=[[random.random() for i in range(len(inp[row])+1)] for j in range(10)]
        for i in range(10):
            for j in range(s2):
                del_weights_l3[i][j+1]=-1*(out[row][i]-y[i])*(y[i])*(1-y[i])*o1[j]
            del_weights_l3[i][0]=-1*(out[row][i]-y[i])*(y[i])*(1-y[i])        
        #update weights at layer 2
        for i in range(10):
            for j in range(s2):
                weights_l3[i][j]=weights_l3[i][j]-(0.75)*del_weights_l3[i][j]

    
        #adjust weights l2
        del_weights_l2=[[random.random() for i in range(len(inp[row])+1)] for j in range(len(inp[row]))]
        for i in range(s1):
            k=0
            for j in range(10):
                k+=-1*(out[row][j]-y[j])*(y[j])*(1-y[j])*weights_l3[j][i+1]
            for j in range(s1):
                del_weights_l2[i][j+1]=o[j]*o1[i]*(1-o1[i])*k
        #update weights at layer 2
        for i in range(s1):
            for j in range(s1):
                weights_l2[i][j]=weights_l2[i][j]-(0.9)*del_weights_l2[i][j]
        

        #adjust weights l1
        del_weights_l1=[[random.random() for i in range(len(inp[row])+1)] for j in range(len(inp[row]))]
        for i in range(s1):
            k=0
            for j in range(10):
                k+=-1*(out[row][j]-y[j])*(y[j])*(1-y[j])*weights_l2[j][i+1]
            for j in range(s1):
                del_weights_l1[i][j+1]=inp[row][j]*o[i]*(1-o[i])*k
        #update weights at layer 1
        for i in range(s1):
            for j in range(s1):
                weights_l1[i][j]=weights_l1[i][j]-(1)*del_weights_l1[i][j]

        
    #print the accuracy
    print(count)
        
if __name__=="__main__":
    main()
