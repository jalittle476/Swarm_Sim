import pygame
import sys

pygame.init()
size = (400, 300)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pygame Test")

clock = pygame.time.Clock()
carryOn = True
while carryOn:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False
    screen.fill((0, 120, 255))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
