import pyximport
from typing import Tuple

pyximport.install(language_level=3)
import argparse
import http.server
import json
import socketserver
import threading
import time
import shutil
import os
import hanabi
import Agents
import random
import hashlib
import pickle
import sys
import traceback
from cgi import parse_header, parse_multipart
from urllib.parse import parse_qs

from Agents.player import Player

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--host", type=str, default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=31337)
args = parser.parse_args()

HOST_NAME = args.host
PORT_NUMBER = args.port

HAND = 0
TRASH = 1
BOARD = 2
TRASHP = 3

debug = True


errlog = sys.stdout

template = """

<table width="100%%" style="font-size:14pt;"> <tr>

<td width="15%%" valign="top"> <br>
<table style="font-size:14pt" width="100%%">
<tr><td> %s </td></tr> <!-- basic info -->

<tr><td> %s </td></tr> <!-- discard -->
</table>
</td>

<td>
<center>
<div> %s </div> <!-- other hand -->
<div> %s </div> <!-- board -->
<div> %s </div> <!-- your hand -->
</center>
</td>

<td width="15%%" valign="top">
%s
</td> <!-- actions -->

</tr> </table>
<a href="/rules" target="_blank" rel="noreferrer noopener" style="font-size: 21px">Rules</a>
"""

info_template = """
%s
<table width="100%%" style="font-size:14pt; border:solid 5px %s;">
  <tr><td width="85%%"><b>Hint tokens left:</b></td><td> %s</td></tr>
  <tr><td><b>Mistakes made so far:</b></td><td> %s</td></tr>
  <tr><td><b>Cards left in deck:</b></td><td> %s</td></tr>
</table>
"""

discard_template = """
%s
<table width="100%%" style="font-size:14pt; border:solid 5px %s;">
<tr><td><center><h2>Discarded</h2></center></td></tr>
<tr><td> %s </td></tr>
</table>
"""

hand_template = """
%s
<table style="border:solid 5px %s; ">
<tr> <td colspan="5"><center><h2>%s</h2></center></td></tr>
<tr>
  <td>%s<br/>%s</td>
  <td>%s<br/>%s</td>
  <td>%s<br/>%s</td>
  <td>%s<br/>%s</td>
  <td>%s<br/>%s</td>
</tr>
</table>
"""

board_template = """
%s
<table style="border:solid 5px %s; ">
<tr> <td colspan="5"><center>%s</center></td> </tr>
<tr>
  <td>%s</td>
  <td>%s</td>
  <td>%s</td>
  <td>%s</td>
  <td>%s</td>
</tr>
</table>
"""

action_template = """
%s
<table style="border:solid 5px %s; ">
  <tr><td><center><h2>Actions</h2></center><br></td></tr>
  <tr><td><div style="font-size:14pt">%s</div></td></tr>
</table>
"""


def format_info(hint, hits, cards, highlight=False, message=""):
    return info_template % (
        message,
        "#FF0000" if highlight else "#FFFFFF",
        hint,
        hits,
        cards,
    )


def format_hand(cards, title, highlight=False, message=""):
    return hand_template % (
        message,
        "#FF0000" if highlight else "#FFFFFF",
        title,
        *cards,
    )


def format_board(game, show, gid):
    if not game.started:
        return (
            '<table><tr><td colspan="5"><center><h1><a href="/gid%s/start/">Start Game</a></h1></center></td></tr><table>'
            % gid
        )
    title = "<h2>Board</h2>"
    if game.done():
        title = '<h2>Game End<h2>Points: {}<br/><a href="/postsurvey/{}">Please complete Post-Game Survey Here</a>'.format(
            game.score(), gid
        )

    def make_board_image(card_with_index):
        (i, card) = card_with_index
        return make_card_image(card, [], (BOARD, 0, i) in show)

    boardcards = list(map(make_board_image, enumerate(game.board)))
    return board_template % ("", "#FFFFFF", title, *boardcards)


def format_all_actions(actions, highlight=False, message=""):
    if isinstance(actions, list):
        return action_template % (
            message,
            "#FF0000" if highlight else "#FFFFFF",
            "\n".join(format_action(x) for x in actions[:15]),
        )
    else:
        return action_template % (
            message,
            "#FF0000" if highlight else "#FFFFFF",
            actions,
        )


def format_action(action_with_meta):
    (i, (action, pnr, card)) = action_with_meta
    result = "You "
    other = "the AI"
    otherp = "their"
    if pnr == 0:
        result = "AI "
        other = "you"
        otherp = "your"
    if i <= 1:
        result += " just "

    if action.type == hanabi.PLAY:
        result += " played <b>" + hanabi.format_card(card) + "</b>"
    elif action.type == hanabi.DISCARD:
        result += " discarded <b>" + hanabi.format_card(card) + "</b>"
    else:
        result += " hinted %s about all %s " % (other, otherp)
        if action.type == hanabi.HINT_COLOR:
            result += hanabi.COLORNAMES[action.col] + " cards"
        else:
            result += str(action.num) + "s"
    if i in {1, 0}:
        return result + "<br><br>"
    return '<div style="color: gray;">' + result + "</div>"


def format_trash(trash, show=[], highlight=False, message=""):
    trash.sort()
    foundtrash = []
    discarded = {}
    trashhtml = '<table width="100%%" style="border-collapse: collapse"><tr>\n'
    for i, c in enumerate(hanabi.ALL_COLORS):
        style = "border-bottom: 1px solid #000"
        if i > 0:
            style += "; border-left: 1px solid #000"
        trashhtml += (
            '<td valign="top" align="center" style="%s" width="20%%">%s</td>\n'
            % (style, hanabi.COLORNAMES[c])
        )
        discarded[c] = []
        for (col, num) in trash:
            if col == c:
                if (
                    (TRASH, 0, -1) in show
                    and (col, num) == trash[-1]
                    and (col, num) not in foundtrash
                ):
                    foundtrash.append((col, num))
                    discarded[c].append('<div style="color: red;">%d</div>' % (num))
                elif (
                    (TRASH, 0, -2) in show
                    and (col, num) == trash[-2]
                    and (col, num) not in foundtrash
                ):
                    foundtrash.append((col, num))
                    discarded[c].append('<div style="color: red;">%d</div>' % (num))
                else:
                    discarded[c].append("<div>%d</div>" % num)
        discarded[c].sort()
    trashhtml += '</tr><tr style="height: 150pt">\n'
    for i, c in enumerate(hanabi.ALL_COLORS):
        style = ' style="vertical-align:top"'
        if i > 0:
            style = ' style="border-left: 1px solid #000; vertical-align:top"'
        trashhtml += '<td valigh="top" align="center" %s>%s</td>\n' % (
            style,
            "\n".join(discarded[c]),
        )
    trashhtml += "</tr></table><br/>"
    if foundtrash:
        trashhtml += 'Cards written in <font color="red">red</font> have been discarded or misplayed since your last turn.'
    return discard_template % (message, "#FF0000" if highlight else "#FFFFF", trashhtml)


def show_game_state(game, player, turn, gid):
    def make_ai_card(card_with_index, highlight):

        (i, (col, num)) = card_with_index
        hintlinks = [
            ("Hint Rank", "/gid%s/%d/hintrank/%d" % (gid, turn, i)),
            ("Hint Color", "/gid%s/%d/hintcolor/%d" % (gid, turn, i)),
        ]
        if game.hints == 0 or game.done() or not game.started:
            hintlinks = []
            highlight = False
        return make_card_image((col, num), hintlinks, highlight)

    aicards = []
    for i, c in enumerate(game.hands[0]):
        aicards.append(make_ai_card((i, c), (HAND, 0, i) in player.show))
        aicards.append(", ".join(player.aiknows[i]))
    while len(aicards) < 10:
        aicards.append("")
    aicards = format_hand(aicards, "Other Player")

    def make_your_card(card_with_index, highlight):
        (i, (col, num)) = card_with_index
        playlinks = [
            ("Play", "/gid%s/%d/play/%d" % (gid, turn, i)),
            ("Discard", "/gid%s/%d/discard/%d" % (gid, turn, i)),
        ]
        if game.done() or not game.started:
            playlinks = []
        return unknown_card_image(playlinks, highlight)

    yourcards = []
    for i, c in enumerate(game.hands[1]):
        if game.done():
            yourcards.append(make_ai_card((i, c), False))
        else:
            yourcards.append(make_your_card((i, c), (HAND, 1, i) in player.show))
        yourcards.append(", ".join(player.knows[i]))
    while len(yourcards) < 10:
        yourcards.append("")
    yourcards = format_hand(yourcards, "You")

    board = format_board(game, player.show, gid)
    localtrash = game.trash[:]
    trash = format_trash(localtrash, player.show)
    actions = format_all_actions(list(enumerate(reversed(player.actions))))

    hints = game.hints
    if hints == 0:
        hints = '<div style="font-weight: bold; font-size: 20pt">0</div>'
    mistakes = 3 - game.hits
    if mistakes == 2:
        mistakes = '<div style="font-weight: bold; font-size: 20pt; color: red">2</div>'
    cardsleft = len(game.deck)
    if cardsleft < 5:
        cardsleft = (
            '<div style="font-weight: bold; font-size: 20pt">%d</div>' % cardsleft
        )
    info = format_info(hints, mistakes, cardsleft)

    return template % (
        info,
        trash,
        aicards,
        board,
        yourcards,
        actions,
    )


def make_circle(x, y, col):
    x += random.randint(-5, 5)
    y += random.randint(-5, 5)
    r0 = random.randint(0, 180)
    r1 = r0 + 360
    result = """
    <circle cx="%f" cy="%d" r="10" stroke="%s" stroke-width="4" fill="none">
       <animate attributeName="r" from="1" to="22" dur="2s" repeatCount="indefinite"/>
       <animate attributeName="stroke-dasharray" values="32, 32; 16, 16; 8,8; 4,4; 2,6; 1,7;" dur="2s" repeatCount="indefinite" calcMode="discrete"/>
       <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="%f %f %f" to="%f %f %f" dur="2s" begin="0s" repeatCount="indefinite"/>
    </circle>
    """
    return result % (x, y, col, r0, x, y, r1, x, y)


def make_card_image(card, links=[], highlight=False):
    (col, num) = card
    image = """
<svg version="1.1" width="125" height="160" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <rect width="125" height="160" x="0" y="0" fill="#66ccff"%s/>
    <text x="8" y="24" fill="%s" font-family="Arial" font-size="24" stroke="black">%s</text>
    <text x="50" y="24" fill="%s" font-family="Arial" font-size="24" stroke="black">%s</text>
    %s
    %s
    <text x="108" y="155" fill="%s" font-family="Arial" font-size="24" stroke="black">%s</text>
</svg>
"""
    ly = 130
    linktext = ""
    for (text, target) in links:
        linktext += """<a class="move" xlink:href="%s">
                           <text x="8" y="%d" fill="blue" font-family="Arial" font-size="12" text-decoration="underline">%s</text>
                       </a>
                       """ % (
            target,
            ly,
            text,
        )
        ly += 23
    l = 35  # left
    r = 90  # right
    c = 62  # center (horizontal)

    t = 45  # top
    m = 70  # middle (vertical)
    b = 95  # bottom
    circles = {
        0: [],
        1: [(c, m)],
        2: [(l, t), (r, b)],
        3: [(l, b), (r, b), (c, t)],
        4: [(l, b), (r, b), (l, t), (r, t)],
        5: [(l, b), (r, b), (l, t), (r, t), (c, m)],
    }
    circ = "\n".join(
        [make_circle(x_y[0], x_y[1], hanabi.COLORNAMES[col]) for x_y in circles[num]]
    )
    highlighttext = ""
    if highlight:
        highlighttext = ' stroke="red" stroke-width="8"'
    return image % (
        highlighttext,
        hanabi.COLORNAMES[col],
        str(num),
        hanabi.COLORNAMES[col],
        hanabi.COLORNAMES[col],
        circ,
        linktext,
        hanabi.COLORNAMES[col],
        str(num),
    )


def unknown_card_image(links=[], highlight=False):
    image = """
<svg version="1.1" width="125" height="160" xmlns="http://www.w3.org/2000/svg">
    <rect width="125" height="160" x="0" y="0" fill="#66ccff"%s/>
    %s
    <text x="35" y="90" fill="black" font-family="Arial" font-size="100">?</text>
</svg>
"""
    ly = 130
    linktext = ""
    for (text, target) in links:
        linktext += """<a class="move" xlink:href="%s">
                           <text x="8" y="%d" fill="blue" font-family="Arial" font-size="12" text-decoration="underline">%s</text>
                       </a>
                       """ % (
            target,
            ly,
            text,
        )
        ly += 23
    highlighttext = ""
    if highlight:
        highlighttext = ' stroke="red" stroke-width="4"'
    return image % (highlighttext, linktext)


gameslock = threading.Lock()
games = {}
participantslock = threading.Lock()
participants = {}
participantstarts = {}


class HTTPPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.actions = []
        self.knows = [set() for i in range(5)]
        self.aiknows = [set() for i in range(5)]
        self.show = []

    def inform(self, action, player, game, model):
        self.show = []
        card = None
        if action.type in [hanabi.PLAY, hanabi.DISCARD]:
            card = game.get_card_changed()
        self.actions.append((action, player, card))
        if player != self.pnr:
            if action.type == hanabi.HINT_COLOR:
                for i, (col, num) in enumerate(game.hands[self.pnr]):
                    if col == action.col:
                        self.knows[i].add(hanabi.COLORNAMES[col])
                        self.show.append((HAND, self.pnr, i))
            elif action.type == hanabi.HINT_NUMBER:
                for i, (col, num) in enumerate(game.hands[self.pnr]):
                    if num == action.num:
                        self.knows[i].add(str(num))
                        self.show.append((HAND, self.pnr, i))
        else:
            if action.type == hanabi.HINT_COLOR:
                for i, (col, num) in enumerate(game.hands[action.pnr]):
                    if col == action.col:
                        self.aiknows[i].add(hanabi.COLORNAMES[col])
                        self.show.append((HAND, action.pnr, i))
            elif action.type == hanabi.HINT_NUMBER:
                for i, (col, num) in enumerate(game.hands[action.pnr]):
                    if num == action.num:
                        self.aiknows[i].add(str(num))
                        self.show.append((HAND, action.pnr, i))

        if action.type == hanabi.DISCARD:
            self.show.append((TRASH, 0, -1))

        elif action.type == hanabi.PLAY:
            (col, num) = card
            if game.board[col][1] == num:
                self.show.append((BOARD, 0, col))
            else:
                self.show.append((TRASH, 0, -1))
        if player == self.pnr and action.type in [hanabi.PLAY, hanabi.DISCARD]:
            del self.knows[action.cnr]
            self.knows.append(set())
        if player != self.pnr and action.type in [hanabi.PLAY, hanabi.DISCARD]:
            del self.aiknows[action.cnr]
            self.aiknows.append(set())


def format_score(sc):
    if sc is None:
        return "not finished"
    return "%d points" % sc


# AIClasses
pool_path = "Agents/configs/players.json"
player_pool = Agents.PlayerPool("AI", 0, pool_path)
json_pool = json.load(open(pool_path))
global_info = {}


class MyHandler(http.server.BaseHTTPRequestHandler):
    def __init__(
        self,
        request: bytes,
        client_address: Tuple[str, int],
        server: socketserver.BaseServer,
    ):
        super().__init__(request, client_address, server)
        self.info = None

    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

    def do_GET(s):
        try:
            return s.perform_response()
        except Exception:
            errlog.write(traceback.format_exc())
            errlog.flush()

    def invalid(s, gid):
        if len(gid) != 16:
            return True
        for c in gid:
            if c not in "0123456789abcdef":
                return True
        if not os.path.exists("log/game%s.log" % gid):
            return True
        return False

    def write_game(s, gid):
        s.wfile.write(b"<html><head><title>Hanabi</title></head>")
        s.wfile.write(b"<body>")
        if os.path.exists(gid + ".temp"):
            with open(gid + ".temp", "rb") as f:
                game_html = f.read()
                s.wfile.write(game_html)
                f.close()
        s.wfile.write(
            bytes(
                """
        <style>
        .loader {
              position: absolute;
              top: 600px;
              right: 200px;
              border: 12px solid #f3f3f3; /* Light grey */
              border-top: 12px solid #3498db; /* Blue */
              border-radius: 50%;
              width: 120px;
              height: 120px;
              animation: spin 1.5s linear infinite;
            }

            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loader"></div>
        """,
                "utf-8",
            )
        )
        s.wfile.write(b"</body></html>")

    def perform_response(s):
        """Respond to a GET request."""
        global games

        game = None
        player = None
        turn = None
        gid = None
        path = s.path
        if s.path.startswith("/gid"):
            gid = s.path[4:20]
            gameslock.acquire()
            if gid in games:
                game, player, turn = games[gid]
            gameslock.release()
            path = s.path[20:]

        if s.path.startswith("/read"):
            s.send_response(200)
            s.send_header("Content-type", "text/html")
            s.end_headers()
            # sends gid to the path
            s.write_game(s.path.split("/")[3])
            return

        if s.path.startswith("/play"):
            s.send_response(200)
            s.send_header("Content-type", "text/html")
            s.end_headers()
            game_url = "/new" + s.path[5:]
            s.wfile.write(
                bytes(
                    """
                <html style="width: 100%; height: 100%; margin: 0; padding: 0">
                <body style="width: 100%; height: 100%; margin: 0; padding: 0">
                <div id="testblock" style="display: flex; width: 100%; height: 100%; flex-direction: column;
                background-color: white; overflow: hidden;">
                <iframe id="loadframe" src="/read"""
                    + s.path[5:]
                    + """" style='flex-grow: 1; border:none; margin: 0; padding: 0;'></iframe>
                </div>
                <div style="display: flex; width: 100%; height: 100%; flex-direction: column;
                background-color: white; overflow: hidden;">
                <iframe id="game_frame" onload="hidePrev()" src='"""
                    + game_url
                    + """' style='flex-grow: 1; border:none; margin: 0; padding: 0;'></iframe>
                </div>
                <script type="text/javascript">
                function hidePrev() {
                    console.log("calling hideprev");
                    var iframe = document.getElementById("game_frame");
                    if (iframe == null) {
                        // iframe hasn't loaded yet, try again in 500ms
                        window.setTimeout(hidePrev, 500);
                        console.log("frame not loaded yet");
                    } else {
                        console.log("frame loaded");
                        // iframe.contentDocument.addEventListener('click',
                        //                 function (e) {
                        //     console.log("calling showprev");
                        //     showPrev();
                        // }, true);
                        var links = iframe.contentDocument.getElementsByClassName("move");
                        for (var i = 0; i < links.length; i++) {
                            links[i].addEventListener('click', function (e) {
                                console.log("calling showprev");
                                showPrev();
                            });
                        }
                        // hide the other frame
                        document.getElementById("testblock").style.display = "none";
                        // window.setTimeout(showPrev, 3000);
                    }
                }

                function showPrev() {
                    document.getElementById("loadframe").src = document.getElementById("loadframe").src;
                    document.getElementById("testblock").style.display = "flex";
                }
                // var iframe = document.getElementById("game_frame");
                // iframe.contentDocument.addEventListener('click',
                //                         function (e) {
                //     console.log("calling showprev");
                //     showPrev();
                // }, true);

                window.addEventListener('beforeunload',
                                        function (e) {
                    var frame_content = document.getElementById("game_frame").contentWindow.document.body.innerHTML;
                    // the frames we *don't* want them to close
                    if (frame_content.search("Actions") != -1 || frame_content.search("Start") != -1
                        || frame_content.search("rate the play skill") != -1) {
                        var message = "You have not finished the study. Are you sure you want to leave?";
                        e.preventDefault();
                        e.returnValue = message;
                        return message;
                    }
                });
                </script>
                </body>
                </html>
                """,
                    "utf-8",
                )
            )
            return

        if s.path == "/hanabiui.png":
            f = open("hanabiui.png", "rb")
            s.send_response(200)
            s.send_header("Content-type", "image/png")
            s.end_headers()
            shutil.copyfileobj(f, s.wfile)
            f.close()
            return

        if s.path.startswith("/favicon"):
            s.send_response(200)
            s.end_headers()
            return

        if s.path.startswith("http://"):
            s.send_response(400)
            s.end_headers()
            return

        if s.path.startswith("/robots.txt"):
            s.send_response(200)
            s.send_header("Content-type", "text/plain")
            s.end_headers()
            s.wfile.write(b"User-agent: *\n")
            s.wfile.write(b"Disallow: /\n")
            return

        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

        if s.path.startswith("/rules"):
            with open("rules.html", "rb") as f:
                doc = f.read()
                s.wfile.write(doc)
                f.close()

        if s.path.startswith("/tutorial-start"):
            _, _, gid = s.path.split("/")
            s.wfile.write(b"<html><head><title>Hanabi</title></head>")
            s.wfile.write(b"<body>")
            instruction = """Your partner's hand will be shown here. You can click on "Hint Rank" / "Hint Color" to give a hint <br>of the rank/color shown on the card. For this tutorial, try to hint the 1's to your partner."""

            links = [[("Hint Rank", s.path), ("Hint Color", s.path)] for i in range(5)]
            links[-2] = [
                ("Hint Rank", "/tutorial-board/{}".format(gid)),
                ("Hint Color", s.path),
            ]
            aicards = [(1, 2), (0, 4), (4, 3), (0, 1), (2, 3)]
            aicards = [
                make_card_image(cd, lks, False) for cd, lks in zip(aicards, links)
            ]
            aicards = sum((list(x) for x in zip(aicards, [""] * 5)), [])
            aicards = format_hand(aicards, "Other Player", True, instruction)
            board = [make_card_image((i, 0), [], False) for i in range(5)]
            board = board_template % ("", "#FFFFFF", "<h2>Board</h2", *board)
            links = [[("Play", s.path), ("Discard", s.path)] for i in range(5)]
            yourcards = [unknown_card_image(lks, False) for lks in links]
            yourcards = sum((list(x) for x in zip(yourcards, [""] * 5)), [])
            yourcards = format_hand(yourcards, "You", False, "")
            trash = format_trash([], [])
            info = format_info(8, 0, 40)
            actions = format_all_actions("")

            s.wfile.write(
                (
                    template
                    % (
                        info,
                        trash,
                        aicards,
                        board,
                        yourcards,
                        actions,
                    )
                ).encode()
            )

            s.wfile.write(b"</body></html>")
            return

        elif s.path.startswith("/tutorial-board"):
            _, _, gid = s.path.split("/")
            s.wfile.write(b"<html><head><title>Hanabi</title></head>")
            s.wfile.write(b"<body>")
            instruction_other = """Now that your partner played green 1, a new card is dealt from the right"""
            instruction_board = """<br>The board will be shown here. Your partner's successful play will be highlighted in red.<br><a href="/tutorial-yourhand/{}"> Click Here </a> to continue""".format(
                gid
            )
            instruction_info = """Basic information will be reflected here."""
            instruction_action = """Recent action history will be displayed here."""

            links = [[("Hint Rank", s.path), ("Hint Color", s.path)] for i in range(5)]
            aicards = [(1, 2), (0, 4), (4, 3), (2, 3), (3, 5)]
            aicards = [
                make_card_image(cd, lks, False) for cd, lks in zip(aicards, links)
            ]
            aicards = sum((list(x) for x in zip(aicards, [""] * 5)), [])
            aicards = format_hand(aicards, "Other Player", True, instruction_other)
            board = [make_card_image((0, 1), [], True)]
            board += [make_card_image((i, 0), [], False) for i in range(1, 5)]
            board = board_template % (
                instruction_board,
                "#FF0000",
                "<h2>Board</h2",
                *board,
            )
            links = [[("Play", s.path), ("Discard", s.path)] for i in range(5)]
            yourcards = [unknown_card_image(lks, False) for lks in links]
            yourcards = sum((list(x) for x in zip(yourcards, [""] * 5)), [])
            yourcards = format_hand(yourcards, "You", False, "")
            trash = format_trash([], [])
            info = format_info(7, 0, 39, True, instruction_info)
            actions = format_all_actions(
                "AI just played green 1", True, instruction_action
            )

            s.wfile.write(
                (template % (info, trash, aicards, board, yourcards, actions)).encode()
            )

            s.wfile.write(b"</body></html>")
            return

        elif s.path.startswith("/tutorial-yourhand"):
            _, _, gid = s.path.split("/")
            s.wfile.write(b"<html><head><title>Hanabi</title></head>")
            s.wfile.write(b"<body>")
            instruction_other = """Suppose your partner did not play the card immediately. Your past hints will be shown beneath the card."""
            instruction_your = """Your hand will be shown here. If your partner hints at you, it will be shown beneath the cards. <br>You can play or discard your cards by clicking the corresponding link. Now try to discard the 4."""

            links = [[("Hint Rank", s.path), ("Hint Color", s.path)] for i in range(5)]
            aicards = [(1, 2), (0, 4), (4, 3), (0, 1), (2, 3)]
            aicards = [
                make_card_image(cd, lks, False) for cd, lks in zip(aicards, links)
            ]
            aicards = sum((list(x) for x in zip(aicards, [""] * 5)), [])
            aicards[-3] = "1"
            aicards = format_hand(aicards, "Other Player", True, instruction_other)
            board = [make_card_image((i, 0), [], False) for i in range(5)]
            board = board_template % ("", "#FFFFFF", "<h2>Board</h2", *board)
            links = [[("Play", s.path), ("Discard", s.path)] for i in range(5)]
            links[0] = [
                ("Play", s.path),
                ("Discard", "/tutorial-discard/{}".format(gid)),
            ]
            yourcards = [unknown_card_image(lks, False) for lks in links]
            yourcards = sum((list(x) for x in zip(yourcards, [""] * 5)), [])
            yourcards[1] = "4"
            yourcards = format_hand(yourcards, "You", True, instruction_your)
            trash = format_trash([], [])
            info = format_info(6, 0, 39)
            actions = format_all_actions("")

            s.wfile.write(
                (
                    template
                    % (
                        info,
                        trash,
                        aicards,
                        board,
                        yourcards,
                        actions,
                    )
                ).encode()
            )

            s.wfile.write(b"</body></html>")
            return

        elif s.path.startswith("/tutorial-discard"):
            _, _, gid = s.path.split("/")
            s.wfile.write(b"<html><head><title>Hanabi</title></head>")
            s.wfile.write(b"<body>")
            instruction = """The discard pile will be shown here. <a href="/tutorial-end/{}"> Click Here </a> to continue """.format(
                gid
            )

            links = [[("Hint Rank", s.path), ("Hint Color", s.path)] for i in range(5)]
            aicards = [(1, 2), (0, 4), (4, 3), (0, 1), (2, 3)]
            aicards = [
                make_card_image(cd, lks, False) for cd, lks in zip(aicards, links)
            ]
            aicards = sum((list(x) for x in zip(aicards, [""] * 5)), [])
            aicards[-3] = "1"
            aicards = format_hand(aicards, "Other Player")
            board = [make_card_image((i, 0), [], False) for i in range(5)]
            board = board_template % ("", "#FFFFFF", "<h2>Board</h2", *board)
            links = [[("Play", s.path), ("Discard", s.path)] for i in range(5)]
            links[0] = [
                ("Play", s.path),
                ("Discard", "/tutorial-discard/{}".format(gid)),
            ]
            yourcards = [unknown_card_image(lks, False) for lks in links]
            yourcards = sum((list(x) for x in zip(yourcards, [""] * 5)), [])
            yourcards = format_hand(yourcards, "You")
            trash = format_trash([(3, 4)], [], True, instruction)
            info = format_info(6, 0, 38)
            actions = format_all_actions("")

            s.wfile.write(
                (
                    template
                    % (
                        info,
                        trash,
                        aicards,
                        board,
                        yourcards,
                        actions,
                    )
                ).encode()
            )

            s.wfile.write(b"</body></html>")
            return

        elif s.path.startswith("/tutorial-end"):
            _, _, gid = s.path.split("/")
            agent = random.choice(
                ["ChiefPlayer"] + [str(x) for x in Agents.default_pool_ids]
            )
            redirect = "/play/{}/{}".format(agent, gid)
            s.wfile.write(
                """<html><head><title>Hanabi</title><meta http-equiv="Refresh" content="5; url='{}'" /></head>\n""".format(
                    redirect
                ).encode()
            )
            s.wfile.write(
                b"<body><h1>You have finished the Tutorial!</h1> <p>You should be redirected to the game in 5 seconds. If not, please click here: </p>\n"
            )
            s.wfile.write(
                '<li><a href="{}">Start Game</a></li>\n'.format(redirect).encode()
            )
            s.wfile.write(b"</body></html>")
            return

        if s.path.startswith("/postsurvey/"):
            gid = s.path[12:]
            s.postsurvey(gid)
            return

        elif path.startswith("/new/") and debug:

            _, _, agent_type, gid = s.path.split("/")
            if agent_type == "ChiefPlayer":
                ai = Agents.ChiefPlayer(agent_type, 0, Agents.default_pool_ids)
            else:
                ai = player_pool.from_dict(agent_type, 0, json_pool[agent_type])

            turn = 1
            player = HTTPPlayer("You", 1)
            nr = random.randint(6, 10000)
            t = (agent_type, nr)
            if not os.path.isdir("log/http_games/{}".format(gid)):
                os.makedirs("log/http_games/{}".format(gid))
            game = hanabi.Game(
                [ai, player],
                "log/http_games/{}".format(gid),
                format=1,
                http_player=1,
                print_game=False,
                log_timestamps=True,
            )
            game.treatment = t
            game.ping = time.time()
            game.started = False
            todelete = []
            gameslock.acquire()
            for g in games:
                if games[g][0].ping + 3600 < time.time():
                    todelete.append(g)
            for g in todelete:
                del games[g]
            games[gid] = (game, player, turn)
            gameslock.release()

        if gid is None or game is None or path.startswith("/restart/"):
            if not debug:
                s.wfile.write(b"<html><head><title>Hanabi</title></head>\n")
                s.wfile.write(b"<body><h1>Invalid Game ID</h1>\n")
                s.wfile.write(b"</body></html>")
                return
            if game is not None:
                del game
            gameslock.acquire()
            if gid is not None and gid in games:
                del games[gid]
            else:
                gid = s.getgid()
            gameslock.release()
            participantslock.acquire()
            if gid not in participants:
                if not os.path.isdir("log/http_games/{}".format(gid)):
                    os.makedirs("log/http_games/{}".format(gid))
                participants[gid] = open(
                    "log/http_games/{}/survey.txt".format(gid), "w"
                )
                participantstarts[gid] = time.time()
            participantslock.release()
            s.presurvey(gid)

            s.wfile.write(b"</ul><br/>")
            s.wfile.write(b"</body></html>")
            return

        if path.startswith("/start/"):
            game.single_turn()
            game.started = True

        parts = path.strip("/").split("/")
        if parts[0] == str(turn):
            actionname = parts[1]
            index = int(parts[2])
            action = None
            if actionname == "hintcolor" and game.hints > 0:
                col = game.hands[0][index][0]
                action = hanabi.Action(hanabi.HINT_COLOR, pnr=0, col=col)
            elif actionname == "hintrank" and game.hints > 0:
                nr = game.hands[0][index][1]
                action = hanabi.Action(hanabi.HINT_NUMBER, pnr=0, num=nr)
            elif actionname == "play":
                action = hanabi.Action(hanabi.PLAY, pnr=1, cnr=index)
            elif actionname == "discard":
                action = hanabi.Action(hanabi.DISCARD, pnr=1, cnr=index)

            if action:
                turn += 1
                gameslock.acquire()
                games[gid] = (game, player, turn)
                gameslock.release()
                game.external_turn(action)
                game.single_turn()

        # s.wfile.write(b"<html><head><title>Hanabi</title></head>")
        # s.wfile.write(b"<body>")
        #
        # s.wfile.write(show_game_state(game, player, turn, gid).encode())
        #
        # s.wfile.write(b"</body></html>")
        s.info = game, player, turn, gid
        s.wfile.write(b"<html><head><title>Hanabi</title></head>")
        s.wfile.write(b"<body>")
        game_state_html = show_game_state(game, player, turn, gid).encode()
        with open(gid + ".temp", "wb") as f:
            f.write(game_state_html)
            f.close()
        s.wfile.write(game_state_html)

        s.wfile.write(b"</body></html>")
        if game.done() and gid is not None and gid in games:
            errlog.write(
                "%s game done. Score: %d\n" % (str(game.treatment), game.score())
            )
            errlog.flush()
            game.finish()
            del [gid]

    def add_choice(s, name, question, answers, default=-1):
        s.wfile.write(b"<p>")
        s.wfile.write((question + "<br/>").encode())
        s.wfile.write(('<fieldset id="%s">\n' % name).encode())
        for i, (aname, text) in enumerate(answers):
            if i == default:
                s.wfile.write(
                    (
                        '<input required name="%s" type="radio" value="%s" id="%s%s" checked="checked"/><label for="%s%s">%s</label><br/>\n'
                        % (name, aname, name, aname, name, aname, text)
                    ).encode()
                )
            else:
                s.wfile.write(
                    (
                        '<input required name="%s" type="radio" value="%s" id="%s%s"/><label for="%s%s">%s</label><br/>\n'
                        % (name, aname, name, aname, name, aname, text)
                    ).encode()
                )
        s.wfile.write(b"</fieldset>\n")
        s.wfile.write(b"</p>")

    def add_question(s, name, question):
        s.wfile.write(b"<p>")
        s.wfile.write((question + "<br/>").encode())
        s.wfile.write(b'<input name="%s"/>\n' % name)
        s.wfile.write(b"</p>")

    def presurvey(s, gid, warn=False):
        s.wfile.write(b"<center><h1>Pre-Game Survey</h1>")
        s.wfile.write(b'<table width="600px">\n<tr><td>')
        s.wfile.write(b'<form action="/submitpre" method="POST">')
        s.presurvey_questions()
        s.wfile.write(b"<p>")
        s.wfile.write(('<input name="gid" type="hidden" value="%s"/>\n' % gid).encode())
        s.wfile.write(b'<input type="submit" value="Finish"/>\n')
        s.wfile.write(b"</form></td></tr></table></center>")

    def presurvey_questions(s, answers={}):

        responses = [
            ("10s", "less than 18 years"),
            ("20s", "18-29 years"),
            ("30s", "30-39 years"),
            ("40s", "40-49 years"),
            ("50s", "50-59 years"),
            ("60s", "greater than 60 years"),
            ("na", "Prefer not to answer"),
        ]
        default = -1
        if "age" in answers:
            default = [a_b[0] for a_b in responses].index(answers["age"])
        s.add_choice("age", "What is your age?", responses, default)

        responses = [
            ("male", "Male"),
            ("female", "Female"),
            ("non binary", "Non binary"),
            ("na", "Prefer not to answer"),
        ]
        default = -1
        if "gender" in answers:
            default = [a_b[0] for a_b in responses].index(answers["age"])
        s.add_choice("gender", "What is your gender?", responses, default)

        responses = [
            ("new", "less than once in two months"),
            ("dabbling", "once a month"),
            ("intermediate", "once every two weeks"),
            ("expert", "more than once a week"),
        ]
        default = -1
        if "bgg" in answers:
            default = [a_b1[0] for a_b1 in responses].index(answers["bgg"])
        s.add_choice(
            "bgg",
            "How often do you play board and/or card games?",
            responses,
            default,
        )

        responses = [("yes", "Yes"), ("no", "No")]
        default = -1
        if "gamer" in answers:
            default = [a_b2[0] for a_b2 in responses].index(answers["gamer"])
        s.add_choice(
            "gamer",
            "Do you consider yourself to be a (board) gamer?",
            responses,
            default,
        )

        responses = [
            ("new", "I have never played before participating in this experiment"),
            ("dabbling", "I have played a few (1-10) times"),
            ("intermediate", "I have played multiple (10-50) times"),
            ("expert", "I have played many (&gt; 50) times"),
        ]
        default = -1
        if "exp" in answers:
            default = [a_b3[0] for a_b3 in responses].index(answers["exp"])
        s.add_choice(
            "exp", "How familiar are you with the card game Hanabi?", responses, default
        )

        responses = [
            (
                "never",
                "I have never played before or can't remember when I played the last time",
            ),
            ("long", "The last time I played has been a long time (over a year) ago"),
            (
                "medium",
                "The last time I played has been some time (between 3 months and a year) ago",
            ),
            ("recent", "The last time I played was recent (up to 3 months ago)"),
        ]
        default = -1
        if "recent" in answers:
            default = [a_b4[0] for a_b4 in responses].index(answers["recent"])
        s.add_choice(
            "recent",
            "When was the last time that you played Hanabi before this experiment?",
            responses,
            default,
        )

        responses = [
            ("never", "I never reach the top score, or I have never played Hanabi"),
            (
                "few",
                "I almost never reach the top score (about one in 50 or more games)",
            ),
            ("sometimes", "I sometimes reach the top score (about one in 6-20 games)"),
            ("often", "I often reach the top score (about one in 5 or fewer games)"),
        ]
        default = -1
        if "score" in answers:
            default = [a_b5[0] for a_b5 in responses].index(answers["score"])
        s.add_choice(
            "score",
            "How often do you typically reach the top score of 25 in Hanabi?",
            responses,
            default,
        )

        responses = [("yes", "Yes"), ("no", "No")]
        default = -1
        if "gamer" in answers:
            default = [a_b6[0] for a_b6 in responses].index(answers["publish"])
        s.add_choice(
            "publish",
            "For this study we have recorded your answers to this survey, as well as a log of actions that you performed in the game. We have <b>not</b> recorded your IP address or any other information that could be linked back to you. Do you agree that we make your answers to the survey and the game log publicly available for future research?",
            responses,
            default,
        )

    def postsurvey(s, gid, warn=False):
        s.wfile.write(
            b"<center><h1>Please answer some questions about your experience with the AI</h1>"
        )
        s.wfile.write(b'<table width="600px">\n<tr><td>')
        s.wfile.write(b'<form action="/submitpost" method="POST">')
        s.postsurvey_questions()
        s.wfile.write(b"<p>")
        s.wfile.write(('<input name="gid" type="hidden" value="%s"/>\n' % gid).encode())
        s.wfile.write(b'<input type="submit" value="Finish"/>\n')
        s.wfile.write(b"</form></td></tr></table></center>")

    def postsurvey_questions(s, answers={}):
        responses = [
            ("1", "The AI never takes reasonable actions"),
            ("2", "The AI rarely takes reasonable actions"),
            ("3", "The AI sometimes takes reasonable actions"),
            ("4", "The AI often takes reasonable actions"),
            ("5", "The AI always takes reasonable actions"),
        ]
        default = -1
        s.add_choice(
            "reasonable",
            "Do you think the AI's actions are reasonable?",
            responses,
            default,
        )

        responses = [
            ("1", "I can never predict the AI's action"),
            ("2", "I can rarely predict the AI's action"),
            ("3", "I can sometimes predict the AI's action"),
            ("4", "I can often predict the AI's action"),
            ("5", "I can always predict the AI's action"),
        ]
        default = -1
        s.add_choice(
            "predictable",
            "Do you think you can predict the AI's actions?",
            responses,
            default,
        )

        responses = [
            ("1", "The AI never understand my intention"),
            ("2", "The AI rarely understand my intention"),
            ("3", "The AI sometimes understand my intention"),
            ("4", "The AI often understand my intention"),
            ("5", "The AI always understand my intention"),
        ]
        default = -1
        s.add_choice(
            "interpret",
            "Do you think you the AI can understand your intention?",
            responses,
            default,
        )

        responses = [
            ("vbad", "The AI played very badly"),
            ("bad", "The AI played badly"),
            ("ok", "The AI played ok"),
            ("good", "The AI played well"),
            ("vgood", "The AI played very well"),
        ]
        default = -1
        if "skill" in answers:
            default = [a_b8[0] for a_b8 in responses].index(answers["skill"])
        s.add_choice(
            "skill", "How would you rate the play skill of this AI?", responses, default
        )

        responses = [
            ("vhate", "I really disliked playing with this AI"),
            ("hate", "I somewhat disliked playing with this AI"),
            ("neutral", "I neither liked nor disliked playing with this AI"),
            ("like", "I somewhat liked playing with this AI"),
            ("vlike", "I really liked playing with this AI"),
        ]
        default = -1
        if "like" in answers:
            default = [a_b9[0] for a_b9 in responses].index(answers["like"])
        s.add_choice(
            "like", "How much did you enjoy playing with this AI?", responses, default
        )
        s.wfile.write(
            bytes(
                """
        <p> General Feedback </p>
        <textarea rows="5" cols="150" name="feedback" id="feedback"></textarea>
        """,
                "utf-8",
            )
        )

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers["content-type"])
        if ctype == "multipart/form-data":
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == "application/x-www-form-urlencoded":
            length = int(self.headers["content-length"])
            postvars = parse_qs(self.rfile.read(length), keep_blank_values=1)
        else:
            postvars = {}
        return postvars

    def getgid(s):
        peer = str(s.connection.getpeername()) + str(time.time()) + str(os.urandom(4))
        return hashlib.sha224(peer.encode()).hexdigest()[:16]

    def do_POST(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()
        tvars = s.parse_POST()
        vars = {}
        for v in tvars:
            vars[v] = tvars[v][0]

        if s.path.startswith("/submitpre"):

            gid = s.getgid()
            if b"gid" in vars and vars[b"gid"]:
                gid = vars[b"gid"].decode()
            if vars.get(b"age", b"10s") == b"10s":
                s.wfile.write(b"<html><head><title>Hanabi</title></head>")
                s.wfile.write(b"<body><center>")
                s.wfile.write(
                    b"<h1>Unfortunately you are not old enough for the study. Thank you for your time and participation</h1>"
                )
                s.wfile.write(b"</body></html>")
                return

            if gid not in participants:
                s.wfile.write(b"<html><head><title>Hanabi</title></head>")
                s.wfile.write(b"<body><center>")
                s.wfile.write(
                    b"<h1>Session timed out. Thank you for your time and participation</h1>"
                )
                s.wfile.write(b"</body></html>")
                return
            participantslock.acquire()
            for r in vars:
                print(r, vars[r], file=participants[gid])
            participants[gid].flush()
            participantslock.release()
            # s.wfile.write(
            #     """<html><head><title>Hanabi</title><meta http-equiv="Refresh" content="5; url='{}'" /></head>\n""".format(
            #         redirect
            #     ).encode()
            # )
            # s.wfile.write(
            #     b"<body><h1>Welcome to Hanabi</h1> <p>First let's go through a toturial to get you familiar with the interface. The tutorial should start in 5 seconds. If not, please click here: </p>"
            # )
            s.wfile.write(
                """
                <div style="margin-left: 100px; margin-right: 100px; margin-bottom: 50px;">
                <center>
                <h1> A brief intro to Hanabi </h1>
                <h4> (adopted from hanabi.cards) </h4>
                <a href="/tutorial-start/{gid}">I already know the rules!</a>
                </center>
                <h2> Objective </h2>
                <p>Hanabi is a card game created by Antoine Bauza. It's cooperative, which means that players are not against each other but assemble to reach a common goal. They incarn here distracted pyrotechnists who - by inattention - mixed their powder, wicks and rockets for a large fireworks display. The show will begin soon and the situation is a bit chaotic. They will need to help each other to prevent the show turning to disaster.</p>
                <p>The goal of the pyrotechnics team is to build 5 fireworks, one of each color (white, red, blue, yellow, green) by combining increasing value cards (1,2,3,4,5) of the same color.</p>
                <h2> Setup </h2>
                <p>At the beginning of a game, you will have 8 hint tokens and 3 strike tokens.</p>
                <p>The deck is composed of 50 cards, 10 of each color with numbers 1, 1, 1, 2, 2, 3, 3, 4, 4, 5.</p>
                <p>In a 2 player game, each player will be dealt 5 cards.</p>
                <p>As you will see, players are not allowed to look at their own cards!</p>
                <h2> Playing the game </h2>
                <p>On each player's turn, they take one (and only one) of the three following actions. You are not allowed to pass.</p>
                <ol>
                <li> Give information to another player.</li>
                <li> Discard a card.</li>
                <li> Play a card.</li>
                </ol>
                <p>Players are not allowed to give hints or suggestions on other player's turns!</p>
                <h3> 1. Give information to another player. </h3>
                <p>When you give information, it will remove a hint token. Note: If you have no more hint tokens, you cannot choose to give information and must pick a different action.</p>
                <p>You then give information to a fellow player about the cards in that player's hand by clicking on it. You can tell the player either about one (and only one) color, or one (and only one) value of card.</p>
                <h3> 2. Discard a card </h3>
                <p>Discarding a card returns a hint token. You discard a card from your hand by tapping it. You then draw a new card from the deck and it will be added to your hand.</p>
                <p>Note: If you have all 8 hint tokens, you cannot discard cards and must pick a different action.</p>
                <p>You can consult discarded cards at anytime.</p>
                <h3> 3. Play a card </h3>
                <p>At your turn, to play a card, take a card from your hand and play it. One of two things happen:</p
                <ol>
                <li>If the card begins or adds to a firework, it will be added to that firework pile</li>
                <li>If the card does not add to a firework, it will be discarded the card and add a red strike token</li
                </ol>
                <p>Then you will draw a replacement card from the deck.</p>
                <p>When a player finishes a firework by playing a value 5 card on it, it will return one blue token to the lid of the box as a bonus. If all the blue tokens are in the box lid, you do not get the bonus.</p>
                <h2> End of the game </h2>
                <p> Hanabi can end in three ways: </p>
                <ol>
                <li>If you get the third red token, you lose the game as the display goes up in flames!</li>
                <li>If the team completes all five colors of firework with a value of 5, the team makes a spectacular victorydisplay and obtains the maximum score of 25 points.</li>
                <li>If a player draws the last card from deck, the game is almost over. Each player gets one more turn,including the player who drew the last card. Players cannot draw more cards during these final turns.</li>
                </ol>
                <p>The players then score their performance based on the fireworks they assembled.</p>
                <center>
                <h2> Here's a tutorial that will walk you through our interface </h2>
                <a href="/tutorial-start/{gid}">Start Tutorial</a>
                </center>
                </div>
                """.format(
                    gid=gid
                ).encode()
            )

        elif s.path.startswith("/submitpost"):

            gid = s.getgid()
            if b"gid" in vars and vars[b"gid"]:
                gid = vars[b"gid"].decode()

            if gid not in participants:
                s.wfile.write(b"<html><head><title>Hanabi</title></head>")
                s.wfile.write(b"<body><center>")
                s.wfile.write(
                    b"<h1>Session timed out. Thank you for your time and participation</h1>"
                )
                s.wfile.write(b"</body></html>")
                return
            participantslock.acquire()
            for r in vars:
                print(r, vars[r], file=participants[gid])
            participants[gid].flush()
            participantslock.release()
            s.wfile.write(b"<html><head><title>Hanabi</title></head>")
            s.wfile.write(b"<body><center>")
            s.wfile.write(b"<h1>Thank you for your participation in this study!</h1>")

    def consentform(s):
        s.wfile.write(b"<html><head><title>Hanabi</title></head>\n")
        s.wfile.write(b"<body>\n")
        s.wfile.write((consent.consent).encode())
        s.wfile.write(
            b'<center><font size="12"><a href="/tutorial">By clicking here I agree to participate in the study</a></font></center><br/><br/><br/>'
        )
        s.wfile.write(b"</body></html>")


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    def finish_request(self, request, client_address):
        request.settimeout(30)
        # "super" can not be used because BaseServer is not created from object
        http.server.HTTPServer.finish_request(self, request, client_address)


if __name__ == "__main__":
    server_class = ThreadingHTTPServer
    if not os.path.exists("log/"):
        os.makedirs("log")
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    errlog.write(time.asctime() + " Server Starts - %s:%s\n" % (HOST_NAME, PORT_NUMBER))
    errlog.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    errlog.write(time.asctime() + " Server Stops - %s:%s\n" % (HOST_NAME, PORT_NUMBER))
    errlog.flush()
