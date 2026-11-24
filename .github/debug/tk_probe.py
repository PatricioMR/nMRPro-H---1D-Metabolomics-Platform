"""Temporary: isolate the Linux Tk crash. Usage: python tk_probe.py <case>."""
import sys
import tkinter as tk

case = sys.argv[1]
root = tk.Tk()
print('Tcl', root.tk.eval('info patchlevel'), flush=True)


def photo():
    from PIL import Image, ImageTk
    img = Image.open('utils/Logo_NMRPro-H.png').convert('RGBA').resize((300, 300))
    return ImageTk.PhotoImage(img)


keep = []
if case == 'bare':
    pass
elif case == 'label_photo':
    keep.append(photo())
    tk.Label(root, image=keep[-1]).pack()
elif case == 'iconphoto':
    keep.append(photo())
    root.iconphoto(True, keep[-1])
elif case == 'emoji':
    m = tk.Menu(root, tearoff=0)
    m.add_command(label='📂 Import CSV/TXT')
    tk.Button(root, text='📁 Import Bruker').pack()
    tk.Label(root, text='✅ ok 🔍').pack()
elif case == 'tk_photo':
    keep.append(tk.PhotoImage(file='utils/Logo_NMRPro-H.png'))
    root.iconphoto(True, keep[-1])
    tk.Label(root, image=keep[-1]).pack()


def churn(n=[0]):
    # idle work like the app's: widgets with scroll commands, created and
    # destroyed, so a corrupted heap shows up quickly
    n[0] += 1
    f = tk.Frame(root)
    lb = tk.Listbox(f)
    sb = tk.Scrollbar(f, command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    for i in range(200):
        lb.insert('end', str(i))
    f.pack()
    root.update_idletasks()
    f.destroy()
    if n[0] < 200:
        root.after(10, churn)
    else:
        print('survived', case, flush=True)
        root.destroy()


root.after(100, churn)
root.mainloop()
