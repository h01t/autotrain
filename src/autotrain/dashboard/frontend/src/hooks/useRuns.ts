import { useQuery } from '@tanstack/react-query'
import { fetchRuns } from '../api/client'

export function useRuns() {
  const query = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
    refetchInterval: 30000,
  })
  return {
    ...query,
    refetch: () => query.refetch(),
  }
}
