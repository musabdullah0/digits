import pygame
import sys
from pygame.locals import *
from PIL import Image
from number_reader import Reader

pygame.init()

screen_size = 500
screen = pygame.display.set_mode((screen_size, screen_size+75))
clock = pygame.time.Clock()
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (73, 159, 104)
blue = (134, 187, 216)
dark_blue = (45, 49, 66)
mouse_position = None
dragging_mouse = False
drawing = False
screen.fill(black)


def make_text(font_size, label, color, background_color, x_offset=0, y_offset=0):
    text = pygame.font.SysFont('silomttf', font_size)
    text_surf = text.render(label, 1, color, background_color)
    x = (screen_size-text_surf.get_width())//2 + x_offset
    y = (screen_size-text_surf.get_height())//2 + y_offset
    return screen.blit(text_surf, (x, y))


def read_num():
    # just call the other script for reading numbers
    guessing = True
    rect = pygame.Rect(0, 0, screen_size, screen_size)
    sub = screen.subsurface(rect)
    pygame.image.save(sub, 'number.png')
    reader = Reader('mnist.pt', 'number.png')
    digit_guess = reader.read()
    print(digit_guess)

    circle = pygame.draw.circle(
        screen, green, (screen_size//2, screen_size//2), screen_size//2-50, 0)
    text = "did you guess %s?" % digit_guess

    while guessing:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        make_text(40, text, blue, green)
        y = make_text(40, 'yeah', green, blue, x_offset=-75, y_offset=100)
        n = make_text(40, 'nope', green, blue, x_offset=75, y_offset=100)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = pygame.mouse.get_pos()
            if y.collidepoint(pos) or n.collidepoint(pos):
                guessing = False
                screen.fill(dark_blue)
                break
        pygame.display.update()
    main_loop()


def drawCircle(pos):
    if pos is not None and pos[1] <= screen_size:
        pygame.draw.circle(screen, white, pos, 30, 0)


def intro():
    global drawing
    intro_running = True
    screen.fill(dark_blue)
    while intro_running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        make_text(100, 'digits.ai', blue, None)
        b = make_text(50, 'start', blue, green, x_offset=0, y_offset=150)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = pygame.mouse.get_pos()
            if b.collidepoint(pos):
                intro_running = False
                drawing = True
                screen.fill(dark_blue)
                break

        pygame.display.update()


def main_loop():
    # function was tryna make it's own local mouse_position variable, so I had to point
    # it to the right one (the global one)
    global mouse_position, dragging_mouse, drawing
    running = True

    g = make_text(30, 'guess', blue, green, x_offset=0,
                  y_offset=screen_size//2+55)

    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_position = event.pos
            if mouse_position[1] < screen_size:
                dragging_mouse = True
            elif g.collidepoint(mouse_position):
                read_num()
                drawing = False
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_mouse = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging_mouse:
                mouse_position = event.pos
        if drawing:
            drawCircle(mouse_position)

        pygame.display.update()
        clock.tick(50)


intro()
main_loop()
pygame.quit()
sys.exit()
