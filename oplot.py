import argparse
import matplotlib.pyplot as plt
import numpy as np
import operator
import pypdf
from datetime import datetime, timedelta
from functools import reduce
from itertools import accumulate
from pathlib import Path


DEFAULT_PATH = r"~/Downloads/0531strekk.pdf"
COLOR = "orangered"

CG_CLASS = "136"
ZERO = timedelta()


# --- Util functions

def transpose(x):
    return list(zip(*x))


def subdict(*keys):
    def subdict(obj):
        return dict(zip(keys, operator.itemgetter(*keys)(obj)))
    return subdict


# --- Data extraction

def parse_time(t):
    parts = t.split(":")
    if len(parts) == 2:
        parts = [0] + parts
    hours, minutes, seconds = map(int, parts)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def parse_dirty_time(t):
    return parse_time(t.rstrip("=-+#&@"))

def parse_line(line):
    return list(map(parse_dirty_time, line.split()))


def parse_page(lines):
    _heading, *rest = lines

    n = 0
    while n < len(rest):
        if any(rest[n].startswith(group) for group in ("Damer", "Herrer")):
            group = rest[n]
            n = n+1

            yield {"group": group}
        elif rest[n].startswith("Side:"):
            n = n+1
        elif rest[n].startswith("Beste strekktid for klassen"):
            n = n+3
        else:
            heading, cumulative, splits_class, comparison = rest[n:n+4]
            splits, class_ = splits_class.rsplit(" ", maxsplit=1)  # !!!

            # header
            rank, heading = heading.split(" ", maxsplit=1)
            name, time = heading.rsplit(" ", maxsplit=1)
            time = parse_time(time)

            # times
            cumulative, splits, comparison = map(parse_line,
                (cumulative, splits, comparison))

            assert (cumulative == list(accumulate(splits)))

            n = n+4

            yield {
                "rank": rank,
                "name": name,
                "class": class_,
                "splits": splits,
            }


def scanner(pdf):
    first_page = pdf.pages[0]

    heading, *rest = first_page.extract_text().splitlines()

    area, title, date = heading.rsplit(" ", maxsplit=2)
    date = datetime.strptime(date, "-%d.%m.%Y")

    yield {"area": area, "date": date}
    yield from parse_page(rest)

    for page in pdf.pages[1:]:
        yield from parse_page(page.extract_text().splitlines())


def parser(pdf):
    context = {}

    for entry in scanner(pdf):
        context.update(entry)

        if "splits" in entry:
            yield context.copy()

def read_table(path):
    pdf = pypdf.PdfReader(path)
    tbl = list(parser(pdf))
    return tbl


# --- Table methods


def where(tbl, predicate): return list(filter(predicate, tbl))
def apply(tbl, transform): return list(map(transform, tbl))


def get_people_like(tbl, name):
    return where(tbl, lambda row: name in row["name"].lower())

def get_person(tbl, name):
    return where(tbl, lambda row: row["name"].lower() == name)[0]


def get_person_group(tbl, person):
    return where(tbl, lambda row: row["group"] == person["group"])


# --- Plotting helper functions


def seqdivmod(x, ys):
    def tail(seq): return tuple(seq[1:])
    return reduce(lambda acc, curr: divmod(acc[0], curr) + tail(acc), ys, [x])


def timedelta_formatter(sec, pos):
    hours, minutes, seconds = map(str, seqdivmod(round(sec), [60, 60]))

    if hours == "0":
        return f"{minutes}:{seconds.zfill(2)}"
    else:
        return f"{hours}:{minutes.zfill(2)}:{seconds.zfill(2)}"


def control_formatter(control_alias):
    def fmt(number):
        return control_alias.get(number, number)
    def control_formatter(control, pos):
        return fmt(control)
    return control_formatter


def arrow_formatter(control_alias):
    def fmt(number):
        return control_alias.get(number, number)
    def arrow_formatter(control, pos):
        return f"{fmt(control-1)}→{fmt(control)}"
    return arrow_formatter


# --- Plotting functions


def plot_cumgroup(ax, person, group):
    tax = ax.twinx()
    ax.get_shared_y_axes().join(ax, tax)  # FIXME

    for row in group:
        splits = row["splits"]
        y = list(map(timedelta.seconds.__get__,
            accumulate(splits, initial=ZERO)))
        x = list(range(len(splits)+1))

        if row["name"] == person["name"]:
            style = dict(
                lw=1.2, label=person["name"], color=COLOR,
                zorder=10, marker="o")
        else:
            style = dict(lw=0.8, color="k", marker="o", markersize=2)

        ax.plot(x, y, **style)

    ax.grid(axis="x", color="k")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(control_formatter({0: "S", max(x): "F"}))
    ax.yaxis.set_major_formatter(timedelta_formatter)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, None)

    total = sum(person["splits"], start=ZERO)
    tax.set_yticks([total.seconds])
    tax.set_yticklabels([str(total)])

    ax.legend(loc="upper left")
    ax.set_title("Totaltid")


def plot_splitgroup(ax, person, group):
    box = []
    for row in group:
        splits = row["splits"]
        y = list(map(timedelta.seconds.__get__, splits))
        x = list(range(1, len(splits)+1))

        if row["name"] == person["name"]:
            style = dict(
                lw=1.2, label=person["name"], color=COLOR,
                zorder=10, marker="o")
        else:
            style = dict(lw=0.8, color="k", marker="o", s=2)

        ax.scatter(x, y, **style)
        box.append(y)

    # boxplots
    box = transpose(box)
    box_plot = ax.boxplot(box, showfliers=False)
    for median in box_plot["medians"]:
        median.set_color("k")

    ax.grid(axis="x", color="k")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(arrow_formatter({0: "S", max(x): "F"}))
    ax.yaxis.set_major_formatter(timedelta_formatter)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(0, None)

    ax.legend(loc="upper left")
    ax.set_title("Strekktider")


def plot_group(person, group):
    print(f'Plotter tall for {person["name"]}')
    fig, [lax, rax] = plt.subplots(2)

    context = subdict("area", "group", "date")(person)
    context["date"] = context["date"].strftime(r"%Y-%m-%d")

    fig.suptitle(", ".join(map(str, context.values())))
    plot_cumgroup(lax, person, group)
    plot_splitgroup(rax, person, group)
    plt.show()


# --- User input

def select_index(length):
    while True:
        try:
            ind = int(input("Velg indeks: ")) - 1
            if 0 <= ind < length:
                return ind
        except ValueError:
            pass
        print(f"Ugyldig indeks. Indeks må være tall i 1 ≤ i ≤ {length}")


def ask_person(people, name):
    count = len(people)

    if count > 1:
        print(f"Personer med '{name}' i navnet:")
        for n, person in enumerate(people, start=1):
            print(f'{n}: {person["name"]}')
        person = people[select_index(len(people))]
    elif len(people) == 1:
        person = people[0]
    else:
        person = None

    return person


# --- Options


def parse_args():
    parser = argparse.ArgumentParser(description="Orienteringsplott")
    parser.add_argument("-n", "--name", required=True, help="Løpernavn")
    parser.add_argument("-p", "--path", default=DEFAULT_PATH, type=Path,
        help="Sti til strekktider")

    return parser.parse_args()


# --- Main

def main(options):
    tbl = read_table(options.path.expanduser())

    people = get_people_like(tbl, options.name)
    person = ask_person(people, options.name)

    if person is not None:
        group = get_person_group(tbl, person)
        plot_group(person, group)
    else:
        print(f"Fant ingen personer med '{options.name}' i navnet.")


if __name__ == "__main__":
    main(parse_args())
