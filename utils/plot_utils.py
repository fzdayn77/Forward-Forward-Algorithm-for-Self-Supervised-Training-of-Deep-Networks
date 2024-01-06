import matplotlib as plt

def plot_curves(title="Title", list_1=[], list_1_label="Label 1", 
                list_2=[], list_2_label="Label 2", x_label="X label", y_label="Y label"):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.plot(list_1, label=list_1_label)
    plt.plot(list_2, label=list_2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()