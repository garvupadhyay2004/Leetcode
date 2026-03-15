#include <stdio.h>

int main() {
    int bt[20], wt[20], tat[20], i, n;
    float wtavg = 0, tatavg = 0;

    printf("Enter the number of processes: ");
    scanf("%d", &n);

    for(i = 0; i < n; i++) {
        printf("Enter Burst Time for Process %d: ", i);
        scanf("%d", &bt[i]);
    }

    wt[0] = 0;  // First process has 0 waiting time
    tat[0] = bt[0];

    for(i = 1; i < n; i++) {
        wt[i] = wt[i-1] + bt[i-1];
        tat[i] = wt[i] + bt[i];
    }

    for(i = 0; i < n; i++) {
        wtavg += wt[i];
        tatavg += tat[i];
    }

    printf("\nPROCESS\t\tBURST TIME\tWAITING TIME\tTURNAROUND TIME\n");
    for(i = 0; i < n; i++) {
        printf("P%d\t\t%d\t\t%d\t\t%d\n", i, bt[i], wt[i], tat[i]);
    }

    printf("\nAverage Waiting Time: %.2f", wtavg/n);
    printf("\nAverage Turnaround Time: %.2f\n", tatavg/n);

    return 0;
}
